import unittest
from unittest.mock import patch

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.check_kv_cache import check_forward_equivalence, check_greedy_equivalence


class KVCacheTests(unittest.TestCase):
    def make_model(self, flash_attn=True, num_kv_heads=2):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            model = MiniMindForCausalLM(MiniMindConfig(
                hidden_size=32, num_hidden_layers=2, vocab_size=97,
                num_attention_heads=4, num_key_value_heads=num_kv_heads,
                intermediate_size=64, max_position_embeddings=64,
                dropout=0.0, flash_attn=flash_attn))
        return model.eval()

    def test_tokenwise_logits_match_full_forward_for_attention_backends_and_kv_head_counts(self):
        ids = torch.tensor([[1, 8, 5, 6, 13, 19, 21], [1, 17, 22, 4, 9, 8, 31]])
        for flash in (False, True):
            for kv_heads in (1, 2, 4):
                with self.subTest(flash_attn=flash, num_kv_heads=kv_heads):
                    check_forward_equivalence(self.make_model(flash, kv_heads), ids, [1]*7,
                                              atol=1e-6, rtol=1e-5)

    def test_chunked_continuation_preserves_causality_and_rotary_positions(self):
        ids = torch.tensor([[1, 8, 5, 6, 13, 19, 21, 7, 35, 8, 9]])
        for flash in (False, True):
            with self.subTest(flash_attn=flash):
                check_forward_equivalence(self.make_model(flash), ids, [4, 3, 1, 3],
                                          atol=1e-6, rtol=1e-5)

    def test_left_padded_batch_matches_full_forward_for_real_tokens(self):
        ids = torch.tensor([[0, 0, 0, 0, 1, 8, 9, 5, 6],
                            [0, 0, 1, 5, 6, 9, 7, 3, 8],
                            [1, 4, 5, 6, 8, 9, 7, 3, 8]])
        mask = (ids != 0).long()
        for flash in (False, True):
            for chunks in ([3, 1, 1, 1, 1, 1, 1], [3, 3, 3]):
                with self.subTest(flash_attn=flash, chunks=chunks):
                    check_forward_equivalence(self.make_model(flash), ids, chunks, mask,
                                              atol=1e-6, rtol=1e-5)

    @torch.inference_mode()
    def test_left_padding_does_not_change_unpadded_sequence_logits(self):
        model = self.make_model()
        ids = torch.tensor([[0, 0, 0, 1, 8, 9, 5], [1, 7, 6, 5, 4, 3, 8]])
        mask = (ids != 0).long()
        prefill = model(ids[:, :4], attention_mask=mask[:, :4], use_cache=True)
        cached = model(ids[:, 4:], attention_mask=mask, past_key_values=prefill.past_key_values,
                       use_cache=True).logits
        for index in range(ids.shape[0]):
            unpadded = ids[index][mask[index].bool()].unsqueeze(0)
            expected = model(unpadded).logits[:, -3:]
            torch.testing.assert_close(cached[index:index+1], expected, atol=1e-6, rtol=1e-5)

    @torch.inference_mode()
    def test_same_prefix_cache_can_be_reused_for_different_continuations(self):
        model = self.make_model()
        prefix = torch.tensor([[1, 6, 8, 9]])
        past = model(prefix, use_cache=True).past_key_values
        snapshot = [(key.clone(), value.clone()) for key, value in past]
        for suffix in (torch.tensor([[10, 11, 12]]), torch.tensor([[29, 28, 27]])):
            actual = model(suffix, past_key_values=past, use_cache=True).logits
            expected = model(torch.cat([prefix, suffix], dim=1)).logits[:, -3:]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            for original, saved in zip(past, snapshot):
                for tensor, copied in zip(original, saved):
                    torch.testing.assert_close(tensor, copied, atol=0, rtol=0)

    def test_greedy_generation_matches_and_actually_uses_cache(self):
        ids = torch.tensor([[0, 0, 1, 5], [1, 3, 8, 7]])
        mask = (ids != 0).long()
        for flash in (False, True):
            model = self.make_model(flash)
            with self.subTest(flash_attn=flash), patch.object(model, 'forward', wraps=model.forward) as forward:
                check_greedy_equivalence(model, ids, mask, max_new_tokens=5)
                lengths = [call.args[0].shape[1] for call in forward.call_args_list]
                self.assertEqual(lengths[:5], [4, 1, 1, 1, 1])
                self.assertEqual(lengths[5:], [4, 5, 6, 7, 8])


if __name__ == '__main__':
    unittest.main()
