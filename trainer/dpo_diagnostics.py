"""Structured DPO diagnostics; calculations here do not contribute to training gradients."""

import json
import random
from pathlib import Path

import torch

from scripts.data_splits import prompt_key


class MetricWindow:
    """Pair-weighted means; explicit min/max/count reductions; per-update gradient means."""

    def __init__(self):
        self.values = {}
        self.weights = {}
        self.pairs = 0
        self.batches = 0

    def update(self, metrics, pairs):
        self.pairs += pairs
        self.batches += 1
        for name, value in metrics.items():
            value = value.detach().float() if isinstance(value, torch.Tensor) else torch.tensor(float(value))
            if name.endswith('_min'):
                self.values[name] = torch.minimum(self.values[name], value) if name in self.values else value
            elif name.endswith('_max'):
                self.values[name] = torch.maximum(self.values[name], value) if name in self.values else value
            elif name.endswith('_count'):
                self.values[name] = self.values.get(name, torch.zeros_like(value)) + value
            else:
                weight = 1 if name.startswith('grad_') else pairs
                self.values[name] = self.values.get(name, torch.zeros_like(value)) + value * weight
                self.weights[name] = self.weights.get(name, 0) + weight

    def snapshot(self):
        return {name: (value / self.weights[name] if name in self.weights else value).item()
                for name, value in self.values.items()}


def append_record(path, event, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'event': event, **payload}, ensure_ascii=False, allow_nan=False) + '\n')


@torch.no_grad()
def batch_integrity(x, y, mask, eos_id, pad_id):
    if mask.shape != y.shape or x.shape != y.shape or x.shape[0] % 2:
        raise ValueError('DPO x/y/mask 形状或 chosen/rejected 配对数量不正确')
    half = x.shape[0] // 2
    lengths = mask.sum(1)
    empty = lengths == 0
    nonbinary = (mask != 0) & (mask != 1)
    eos_counts = ((y == eos_id) & mask.bool()).sum(1)
    pad_active = (y == pad_id) & mask.bool()
    starts = mask.argmax(1)
    prefix = torch.arange(x.shape[1], device=x.device)[None] <= starts[:half, None]
    prompt_mismatch = (starts[:half] != starts[half:]) | ((x[:half] != x[half:]) & prefix).any(1)
    counts = {'empty_mask_count': empty.sum(), 'nonbinary_mask_count': nonbinary.sum(),
              'missing_or_multiple_eos_count': (eos_counts != 1).sum(),
              'padding_supervised_count': pad_active.sum(), 'prompt_mismatch_count': prompt_mismatch.sum()}
    if any(bool(value) for value in counts.values()):
        raise ValueError(f'DPO 训练批次不满足数据约束: { {key: int(value) for key, value in counts.items()} }')
    return {**counts,
            'chosen_tokens': lengths[:half].float().mean(),
            'rejected_tokens': lengths[half:].float().mean(),
            'chosen_tokens_min': lengths[:half].min(), 'chosen_tokens_max': lengths[:half].max(),
            'rejected_tokens_min': lengths[half:].min(), 'rejected_tokens_max': lengths[half:].max()}


@torch.no_grad()
def token_kl_metrics(policy_log_probs, ref_log_probs, mask):
    """KL(policy || ref) over vocabulary, on teacher-forced contexts; sequence means."""
    lengths = mask.sum(1).clamp_min(1)
    kl = (policy_log_probs.detach().exp() * (policy_log_probs.detach() - ref_log_probs.detach())).sum(-1)
    per_sequence = (kl * mask).sum(1) / lengths
    half = mask.shape[0] // 2
    return {'chosen_token_kl_policy_ref': per_sequence[:half].mean(),
            'rejected_token_kl_policy_ref': per_sequence[half:].mean()}


def probe_indices(samples, count, seed, keys_file=None):
    if keys_file:
        keys = json.loads(Path(keys_file).read_text())['prompt_keys']
        by_key = {row.get('prompt_key') or prompt_key(row['chosen']): index
                  for index, row in enumerate(samples)}
        if len(keys) != len(set(keys)) or any(key not in by_key for key in keys):
            raise ValueError('固定探针问题缺失或重复')
        return [by_key[key] for key in keys]
    return random.Random(seed).sample(range(len(samples)), min(count, len(samples)))


@torch.inference_mode()
def evaluate_probe(model, ref_model, dataset, indices, tokenizer, device, batch_size,
                   beta, nll_weight, length_normalize, loss_function, log_prob_function, include_kl=True):
    from torch.utils.data._utils.collate import default_collate

    raw_model = model.module if hasattr(model, 'module') else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    previous_training = raw_model.training
    raw_model.eval()
    window = MetricWindow()
    cuda_devices = [torch.device(device).index or torch.cuda.current_device()] if str(device).startswith('cuda') else []
    try:
        # Preserve RNG states so probes cannot change the training stream.
        with torch.random.fork_rng(devices=cuda_devices):
            for start in range(0, len(indices), batch_size):
                batch = default_collate([dataset[index] for index in indices[start:start+batch_size]])
                x = torch.cat([batch['x_chosen'], batch['x_rejected']]).to(device)
                y = torch.cat([batch['y_chosen'], batch['y_rejected']]).to(device)
                mask = torch.cat([batch['mask_chosen'], batch['mask_rejected']]).to(device)
                policy_logits = raw_model(x).logits
                ref_logits = ref_model(x).logits
                policy, policy_full = log_prob_function(policy_logits, y, return_full=True)
                reference, reference_full = log_prob_function(ref_logits, y, return_full=True)
                loss, metrics = loss_function(reference, policy, mask, beta, nll_weight,
                                              length_normalize, return_metrics=True)
                integrity = batch_integrity(x, y, mask, tokenizer.eos_token_id, tokenizer.pad_token_id)
                values = {**metrics, **integrity, 'objective_loss': loss,
                          'policy_reference_logit_difference_max': (policy_logits[mask.bool()]-ref_logits[mask.bool()]).abs().max()}
                if include_kl:
                    values.update(token_kl_metrics(policy_full, reference_full, mask))
                window.update(values, x.shape[0] // 2)
    finally:
        raw_model.train(previous_training)
    return {'pairs': window.pairs, 'metrics': window.snapshot()}
