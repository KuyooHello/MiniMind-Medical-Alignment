import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from trainer.train_dpo import dpo_loss
from scripts.eval_ppl import assistant_token_mask, evaluate_ppl
from scripts.generation_utils import load_tokenizer
from scripts.audit_preferences import quality_flags


class AlignmentObjectiveTests(unittest.TestCase):
    def test_factorial_initial_loss_uses_same_per_token_nll_scale(self):
        ref = torch.tensor([[-2., -2., -99.], [-2., -2., -2.]])
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
        for normalize in (False, True):
            for weight in (0., 0.1):
                value, metrics = dpo_loss(ref, ref, mask, 0.15, weight, normalize, True)
                self.assertAlmostEqual(value.item(), math.log(2) + weight * 2, places=6)
                self.assertAlmostEqual(metrics["chosen_nll"].item(), 2, places=6)

    def test_improving_preference_reduces_each_objective(self):
        ref = torch.full((2, 4), -2.)
        policy = ref.clone()
        policy[0] += 0.2
        policy[1] -= 0.2
        mask = torch.ones_like(ref)
        for normalize in (False, True):
            for weight in (0., 0.1):
                self.assertLess(dpo_loss(ref, policy, mask, .15, weight, normalize),
                                dpo_loss(ref, ref, mask, .15, weight, normalize))

    def test_masked_values_do_not_change_objective_or_gradients(self):
        reference = torch.full((2, 3), -2.)
        policy = reference.clone().requires_grad_()
        mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
        loss = dpo_loss(reference, policy, mask, .15)
        loss.backward()
        self.assertTrue(torch.equal(policy.grad[mask == 0], torch.zeros_like(policy.grad[mask == 0])))

    def test_response_mask_excludes_prompt_and_handles_multiple_turns(self):
        ids = [0, 10, 11, 90, 91, 5, 6, 99, 20, 10, 11, 90, 91, 7, 99]
        mask = assistant_token_mask(ids, [90, 91], [99])
        self.assertEqual([value for value, active in zip(ids, mask) if active], [5, 6, 99, 7, 99])

    def test_response_mask_handles_truncated_answer(self):
        self.assertEqual(assistant_token_mask([1, 2, 90, 91, 4, 5], [90, 91], [99]),
                         [0, 0, 0, 0, 1, 1])

    def test_ppl_is_invariant_to_left_padding_and_batch_size(self):
        tokenizer = load_tokenizer()
        rows = [{"conversations": [{"role": "user", "content": question},
                                   {"role": "assistant", "content": answer}]}
                for question, answer in [("短问题", "短回答"),
                                         ("这是一个明显更长的问题，请解释原因。", "这是较长的参考回答。")]]

        def uniform_model(input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, len(tokenizer)))

        with patch("scripts.eval_ppl.load_model", return_value=uniform_model):
            single = evaluate_ppl("unused.pth", rows, tokenizer, "cpu", batch_size=1)
            batched = evaluate_ppl("unused.pth", rows, tokenizer, "cpu", batch_size=2)
        for key in ("tokens", "response_tokens"):
            self.assertEqual(single[key], batched[key])
        for key in ("loss", "response_loss"):
            self.assertAlmostEqual(single[key], batched[key], places=5)

    def test_preference_filter_does_not_remove_useful_hard_negative(self):
        pair = {"chosen": [{"role": "user", "content": "如何处理不适？"},
                           {"role": "assistant", "content": "先明确原因并接受专业评估。"}],
                "rejected": [{"role": "user", "content": "如何处理不适？"},
                             {"role": "assistant", "content": "可以口服甲醛。"}]}
        self.assertEqual(quality_flags(pair), [])
        pair["chosen"][-1]["content"] = "可以口服甲醛。"
        self.assertIn("reference_known_unsafe_or_wrong_drug_class", quality_flags(pair))


if __name__ == "__main__":
    unittest.main()
