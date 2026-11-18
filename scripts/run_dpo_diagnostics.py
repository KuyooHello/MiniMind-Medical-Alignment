"""Rerun the fixed DPO matrix with structured logs, then audit validation and replication."""

import argparse
import csv
import json
import os
import random
from pathlib import Path

from scripts.data_splits import load_verified_splits, prompt_key, read_jsonl, sha256_file, validate_splits, write_json
from scripts.run_experiments import ROOT, train


def prepare(config):
    if sha256_file(config['initial_checkpoint']) != config['initial_checkpoint_sha256']:
        raise ValueError('初始 SFT 权重与诊断计划不一致')
    splits, _ = load_verified_splits(config['split_dir'])
    datasets = {}
    for path, digest in config['data_sha256'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'诊断数据发生变化: {path}')
        rows = read_jsonl(path)
        validate_splits(splits, rows)
        datasets[path] = {prompt_key(row['chosen']): row for row in rows}
    common = set.intersection(*(set(rows) for rows in datasets.values()))
    keys = random.Random(config['settings']['probe_seed']).sample(sorted(common), config['settings']['probe_samples'])
    for key in keys:
        pairs = [rows[key] for rows in datasets.values()]
        if any(pair != pairs[0] for pair in pairs[1:]):
            raise ValueError('共享探针的 chosen/rejected 文本不一致')
    output_dir = ROOT/config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir/'probe_keys.json'
    payload = {'prompt_keys': keys, 'data_sha256': config['data_sha256'], 'seed': config['settings']['probe_seed']}
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('已有探针与当前计划不一致')
    write_json(path, payload)
    return output_dir, path


def train_matrix(config, directory, keys_path):
    settings = {**config['settings'], 'probe_keys_file': os.path.relpath(keys_path, ROOT/'trainer')}
    for variant in config['variants']:
        train(variant['name'], 'train_dpo.py', settings, variant['overrides'], output_dir=directory)


def compare_state_tensors(current_path, previous_path):
    import torch

    current = torch.load(current_path, map_location='cpu', weights_only=True)
    previous = torch.load(previous_path, map_location='cpu', weights_only=True)
    if set(current) != set(previous):
        raise ValueError('复现检查的 state_dict 键不一致')
    changed = 0
    maximum = 0.0
    for key in current:
        if not torch.equal(current[key], previous[key]):
            changed += 1
            maximum = max(maximum, (current[key].float()-previous[key].float()).abs().max().item())
    return {'state_tensors_identical': changed == 0, 'changed_state_tensors': changed,
            'max_absolute_state_tensor_difference': maximum}


def evaluate_matrix(config_path, config, directory):
    import torch
    from scripts.eval_ppl import evaluate_ppl
    from scripts.generation_utils import load_tokenizer

    splits, _ = load_verified_splits(config['split_dir'])
    rows = splits['val'][:config['validation_samples']]
    tokenizer = load_tokenizer()
    initial = evaluate_ppl(config['initial_checkpoint'], rows, tokenizer, 'cuda')
    previous_val = json.loads(Path('results/validation.json').read_text())
    previous_by_id = {row['id']: row for row in previous_val['results']}
    results, csv_rows = [], []
    health_keys = ('empty_mask_count', 'nonbinary_mask_count', 'missing_or_multiple_eos_count',
                   'padding_supervised_count', 'prompt_mismatch_count', 'nonfinite_loss_count')
    for variant in config['variants']:
        name = variant['name']
        checkpoint = directory/f'{name}_768.pth'
        metadata = json.loads(checkpoint.with_suffix('.training.json').read_text())
        expected_settings = {**config['settings'], **variant['overrides']}
        if any(metadata['settings'].get(key) != value for key, value in expected_settings.items()):
            raise ValueError(f'训练参数与诊断计划不一致: {name}')
        if metadata['checkpoint_sha256'] != sha256_file(checkpoint):
            raise ValueError(f'诊断权重发生变化: {name}')
        for source, digest in metadata['source_sha256'].items():
            if sha256_file(source) != digest:
                raise ValueError(f'训练源码与当前检查版本不一致: {source}')
        diagnostic_path = directory/f'{name}_768.diagnostics.jsonl'
        records = read_jsonl(diagnostic_path)
        if not metadata.get('completed') or any(row['event'] == 'failure' for row in records):
            raise ValueError(f'训练未正常完成: {name}')
        windows = [row for row in records if row['event'] == 'train_window']
        epochs = [row for row in records if row['event'] == 'epoch_end']
        probes = [row for row in records if row['event'] == 'probe']
        expected_pairs = len(read_jsonl(metadata['settings']['data_path'].removeprefix('../'))) * metadata['settings']['epochs']
        if len(epochs) != metadata['settings']['epochs'] or sum(row['pairs'] for row in epochs) != expected_pairs:
            raise ValueError(f'训练覆盖的样本数不正确: {name}')
        if not probes[0].get('initial') or probes[0]['metrics']['policy_reference_logit_difference_max'] != 0:
            raise ValueError(f'初始化策略/参考模型不一致: {name}')
        health = {key: sum(row['metrics'].get(key, 0) for row in epochs) for key in health_keys}
        if any(health.values()):
            raise ValueError(f'诊断发现数据异常: {name}')
        metrics = evaluate_ppl(str(checkpoint), rows, tokenizer, 'cuda')
        previous_path = f"out/experiments/{variant['previous']}_768.pth"
        replication = compare_state_tensors(checkpoint, previous_path)
        replication['previous_validation_response_ppl'] = previous_by_id[variant['previous']]['response_ppl']
        replication['validation_response_ppl_difference'] = metrics['response_ppl'] - replication['previous_validation_response_ppl']
        row = {'id': name, 'previous': variant['previous'], 'checkpoint': str(checkpoint.relative_to(ROOT)),
               'checkpoint_sha256': sha256_file(checkpoint), 'settings': metadata['settings'],
               'diagnostic_sha256': sha256_file(diagnostic_path), 'source_sha256': metadata['source_sha256'],
               'health_counts': health, 'training_pairs_seen': expected_pairs,
               'initial_probe': probes[0], 'final_probe': probes[-1], 'epoch_summaries': epochs,
               'train_windows': len(windows), 'probe_evaluations': len(probes),
               'validation': metrics, 'replication': replication}
        results.append(row)
        for record in records:
            if 'metrics' in record:
                csv_rows.append({'run': name, 'event': record['event'], 'epoch': record['epoch'],
                                 'step': record['step'], 'pairs': record.get('pairs', ''),
                                 'batches': record.get('batches', ''), **record['metrics']})
        print(f"DIAGNOSTIC RESULT {name}: validation response PPL={metrics['response_ppl']:.6f}; "
              f"state tensors identical={replication['state_tensors_identical']}", flush=True)
        torch.cuda.empty_cache()
    fields = ['run', 'event', 'epoch', 'step', 'pairs', 'batches']
    fields += sorted(set.union(*(set(row) for row in csv_rows))-set(fields))
    with Path('results/dpo_diagnostic_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json('results/dpo_diagnostics.json', {
        'purpose': config['purpose'], 'config_sha256': sha256_file(config_path),
        'validation_samples': len(rows), 'validation_data_sha256': sha256_file(Path(config['split_dir'])/'val.jsonl'),
        'initial_checkpoint_sha256': config['initial_checkpoint_sha256'], 'initial_validation': initial,
        'probe_policy': '32 shared retained training pairs; fixed empty-think removal and FP32 eval; same pairs for all variants.',
        'metric_scope': 'Pair-weighted sequence means of masked response-token means; gradient means per optimizer update.',
        'kl_scope': 'Teacher-forced conditional token KL(policy || reference), not full sequence-distribution KL.',
        'evaluation_policy': 'Same-configuration replication; validation only. No test-set evaluation or new model selection.',
        'results': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dpo_diagnostics.json')
    parser.add_argument('--phase', choices=['train', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    directory, keys_path = prepare(config)
    if args.phase in ('train', 'all'):
        train_matrix(config, directory, keys_path)
    if args.phase in ('evaluate', 'all'):
        evaluate_matrix(args.config, config, directory)


if __name__ == '__main__':
    main()
