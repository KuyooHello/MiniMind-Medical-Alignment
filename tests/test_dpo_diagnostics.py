import copy
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dataset.lm_dataset import DPODataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.data_splits import prompt_key, write_json, write_jsonl
from scripts.generation_utils import load_tokenizer
from trainer.dpo_diagnostics import MetricWindow, batch_integrity, evaluate_probe, probe_indices, token_kl_metrics
from trainer.train_dpo import dpo_loss, logits_to_log_probs


class DPODiagnosticsTests(unittest.TestCase):
    def test_existing_diagnostics_are_rejected_before_loading_models_or_writing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostic = root/'existing.jsonl'
            diagnostic.write_text('existing diagnostic\n')
            metadata = root/'sentinel_32.training.json'
            metadata.write_text('{"completed": true}\n')
            project = Path(__file__).resolve().parents[1]
            process = subprocess.run([sys.executable, str(project/'trainer/train_dpo.py'),
                '--save_dir', directory, '--save_weight', 'sentinel', '--hidden_size', '32',
                '--device', 'cpu', '--diagnostic_path', str(diagnostic)],
                cwd=project/'trainer', capture_output=True, text=True)
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertIn('诊断记录已存在', process.stderr)
            self.assertEqual(metadata.read_text(), '{"completed": true}\n')
            self.assertEqual(diagnostic.read_text(), 'existing diagnostic\n')

    def test_window_uses_pair_weights_counts_extremes_and_update_weights(self):
        window = MetricWindow()
        window.update({'loss': .2, 'grad_norm': 2, 'tokens_min': 3, 'tokens_max': 5, 'bad_count': 1}, 3)
        window.update({'loss': 1., 'grad_norm': 4, 'tokens_min': 1, 'tokens_max': 7, 'bad_count': 2}, 1)
        result = window.snapshot()
        self.assertAlmostEqual(result['loss'], .4, places=6)
        self.assertEqual(result['grad_norm'], 3)
        self.assertEqual(result['tokens_min'], 1)
        self.assertEqual(result['tokens_max'], 7)
        self.assertEqual(result['bad_count'], 3)
        self.assertEqual(window.pairs, 4)

    def test_batch_integrity_rejects_empty_masks_eos_padding_and_prompt_errors(self):
        x = torch.tensor([[1, 7, 8, 9, 2], [1, 7, 8, 10, 2]])
        y = torch.tensor([[7, 8, 9, 2, 0], [7, 8, 10, 2, 0]])
        mask = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]])
        self.assertEqual(batch_integrity(x, y, mask, 2, 0)['empty_mask_count'], 0)
        for issue in ('empty', 'eos', 'padding', 'prompt', 'nonbinary'):
            a, b, m = x.clone(), y.clone(), mask.clone()
            if issue == 'empty': m[0] = 0
            if issue == 'eos': b[0, 3] = 9
            if issue == 'padding': m[0, 4] = 1
            if issue == 'prompt': a[1, 1] = 6
            if issue == 'nonbinary': m[0, 1] = 2
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                batch_integrity(a, b, m, 2, 0)

    def test_kl_direction_and_masking_match_known_distributions(self):
        p = torch.tensor([.5, .5]).log().expand(2, 3, 2).clone()
        q = torch.tensor([.25, .75]).log().expand_as(p).clone()
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        q[mask == 0] = torch.tensor([.99, .01]).log()
        metrics = token_kl_metrics(p, q, mask)
        expected = .5*math.log(.5/.25)+.5*math.log(.5/.75)
        for value in metrics.values():
            self.assertAlmostEqual(value.item(), expected, places=6)
        self.assertEqual(token_kl_metrics(p, p, mask)['chosen_token_kl_policy_ref'], 0)

    def test_probe_keys_preserve_cases_after_dataset_reordering(self):
        samples = [{'prompt_key': key} for key in ('a', 'b', 'c')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'keys.json'
            write_json(path, {'prompt_keys': ['c', 'a']})
            self.assertEqual(probe_indices(samples, 2, 73, path), [2, 0])
            self.assertEqual(probe_indices(list(reversed(samples)), 2, 73, path), [0, 2])
            write_json(path, {'prompt_keys': ['missing']})
            with self.assertRaises(ValueError):
                probe_indices(samples, 2, 73, path)

    def test_fixed_probe_preserves_rng_model_mode_and_initial_reference(self):
        tokenizer = load_tokenizer()
        tokenizer.padding_side = 'right'
        pair = {'chosen': [{'role': 'user', 'content': '什么是监督学习？'},
                           {'role': 'assistant', 'content': '使用带标签的样本学习输入与输出的关系。'}],
                'rejected': [{'role': 'user', 'content': '什么是监督学习？'},
                             {'role': 'assistant', 'content': '这是一个需要解释的问题。'}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pairs.jsonl'
            write_jsonl(path, [pair])
            dataset = DPODataset(str(path), tokenizer, max_length=64, remove_empty_think=True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(8)
                model = MiniMindForCausalLM(MiniMindConfig(hidden_size=32, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tokenizer),
                    intermediate_size=64, max_position_embeddings=128, dropout=.2, neftune_noise_alpha=5))
            reference = copy.deepcopy(model).eval().requires_grad_(False)
            python_state = random.getstate()
            torch_state = torch.random.get_rng_state().clone()
            result = evaluate_probe(model, reference, dataset, [0], tokenizer, 'cpu', 1, .15, .1, True,
                                    dpo_loss, logits_to_log_probs)
            self.assertTrue(model.training)
            self.assertFalse(reference.training)
            self.assertEqual(random.getstate(), python_state)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            metrics = result['metrics']
            self.assertEqual(metrics['policy_reference_logit_difference_max'], 0)
            self.assertEqual(metrics['chosen_token_kl_policy_ref'], 0)
            self.assertEqual(metrics['preference_tie_rate'], 1)
            self.assertAlmostEqual(metrics['objective_loss'], math.log(2)+.1*metrics['chosen_nll'], places=6)


if __name__ == '__main__':
    unittest.main()
