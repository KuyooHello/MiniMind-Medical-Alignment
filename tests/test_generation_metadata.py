import unittest

import torch

from scripts.generation_utils import generate_replies, load_tokenizer


class FixedGeneration:
    def __init__(self, continuations):
        self.continuations = torch.tensor(continuations)

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, self.continuations.to(input_ids.device)], dim=1)


class GenerationMetadataTests(unittest.TestCase):
    def test_first_eos_ends_length_even_when_batch_has_repeated_eos(self):
        tokenizer = load_tokenizer()
        model = FixedGeneration([[101, tokenizer.eos_token_id, tokenizer.eos_token_id], [102, 103, 104]])
        replies, metadata = generate_replies(model, tokenizer, ['问题一', '问题二'], 'cpu', batch_size=2,
                                             max_new_tokens=3, return_metadata=True)
        self.assertEqual(len(replies), 2)
        self.assertEqual(metadata[0], {'generated_tokens': 2, 'eos_seen': True, 'generation_cap_without_eos': False})
        self.assertEqual(metadata[1], {'generated_tokens': 3, 'eos_seen': False, 'generation_cap_without_eos': True})
        plain = generate_replies(model, tokenizer, ['问题一', '问题二'], 'cpu', batch_size=2, max_new_tokens=3)
        self.assertEqual(replies, plain)

    def test_eos_at_budget_boundary_is_not_labelled_as_missing_eos(self):
        tokenizer = load_tokenizer()
        model = FixedGeneration([[101, 102, tokenizer.eos_token_id]])
        _, metadata = generate_replies(model, tokenizer, ['问题'], 'cpu', max_new_tokens=3, return_metadata=True)
        self.assertEqual(metadata[0]['generated_tokens'], 3)
        self.assertTrue(metadata[0]['eos_seen'])
        self.assertFalse(metadata[0]['generation_cap_without_eos'])


if __name__ == '__main__':
    unittest.main()
