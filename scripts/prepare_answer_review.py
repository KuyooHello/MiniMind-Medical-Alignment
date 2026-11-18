"""Freeze selected candidates, generate unseen held-out pairs, and hide their identities."""

import argparse
import json
import random
import secrets
from pathlib import Path

import torch

from scripts.blind_review import prepare
from scripts.data_splits import (
    ensure_outputs_available, load_verified_splits, prompt_key, read_jsonl,
    sha256_file, validate_splits, write_json,
)
from scripts.generation_utils import generate_replies, load_model, load_tokenizer, seed_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/blind_review.json')
    parser.add_argument('--output-dir', default='out/answer_blind_review')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    directory = Path(args.output_dir)
    cases_path = directory/'cases.json'
    form_path = directory/'review.csv'
    mapping_path = directory/'mapping.private.json'
    protocol_path = directory/'protocol.json'
    ensure_outputs_available([cases_path, form_path, mapping_path, protocol_path])
    selection = json.loads(Path(config['selection_record']).read_text())
    if sha256_file(config['sft_checkpoint']) != sha256_file(selection['selected_checkpoint']):
        raise ValueError('SFT 权重与冻结的验证集选择不一致')
    if sha256_file(config['dpo_checkpoint']) != sha256_file(selection['best_dpo_checkpoint']):
        raise ValueError('DPO 权重与冻结的验证集选择不一致')
    splits, manifest = load_verified_splits(config['split_dir'])
    if sha256_file(Path(config['split_dir'])/'manifest.json') != selection['manifest_sha256']:
        raise ValueError('数据划分与冻结选择记录不一致')
    training_path = Path(config['dpo_checkpoint']).with_suffix('.training.json')
    training = json.loads(training_path.read_text())
    dpo_path = training['settings']['data_path'].removeprefix('../')
    if sha256_file(dpo_path) != training['data_sha256']:
        raise ValueError('DPO 数据与权重训练记录不一致')
    validate_splits(splits, read_jsonl(dpo_path))
    excluded, exclusion_records = set(), []
    for file in config['exclude_previous_cases']:
        path = Path(file)
        if not path.is_file():
            continue
        previous = json.loads(path.read_text())
        excluded.update(row.get('prompt_key') or prompt_key([{'role': 'user', 'content': row['prompt']}])
                        for row in previous)
        exclusion_records.append({'path': file, 'sha256': sha256_file(path), 'cases': len(previous)})
    candidates = {prompt_key(row['conversations']): row for row in splits[config['split']]}
    eligible_keys = sorted(set(candidates)-excluded)
    if len(eligible_keys) < config['samples']:
        raise ValueError('未展示的留出问题不足')
    keys = random.Random(config['prompt_seed']).sample(eligible_keys, config['samples'])
    prompts = [next(message['content'] for message in candidates[key]['conversations']
                    if message['role']=='user') for key in keys]
    checkpoints = {side: config[f'{side}_checkpoint'] for side in ('sft', 'dpo')}
    protocol = {'config': config, 'config_sha256': sha256_file(args.config),
                'selection_sha256': sha256_file(config['selection_record']),
                'manifest_sha256': sha256_file(Path(config['split_dir'])/'manifest.json'),
                'data_sha256': sha256_file(Path(config['split_dir'])/f"{config['split']}.jsonl"),
                'dpo_data_sha256': sha256_file(dpo_path),
                'checkpoint_sha256': {side: sha256_file(path) for side, path in checkpoints.items()},
                'samples': len(keys), 'eligible_unseen_questions': len(eligible_keys),
                'previously_exposed_questions_in_selected_set': len(set(keys)&excluded),
                'excluded_case_files': exclusion_records, 'prompt_keys': keys,
                'device': args.device, 'generation_dtype': 'bfloat16' if str(args.device).startswith('cuda')
                    and torch.cuda.is_bf16_supported() else ('float16' if str(args.device).startswith('cuda') else 'float32')}
    write_json(protocol_path, protocol)
    tokenizer = load_tokenizer()
    replies, stopping = {}, {}
    for side, checkpoint in checkpoints.items():
        seed_generation(config['generation_seed'])
        model = load_model(checkpoint, args.device)
        replies[side], stopping[side] = generate_replies(
            model, tokenizer, prompts, args.device, return_metadata=True, **config['generation'])
        del model
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    cases = [{'id': index+1, 'prompt_key': key, 'prompt': question,
              'sft_reply': replies['sft'][index], 'dpo_reply': replies['dpo'][index],
              'generation': {side: stopping[side][index] for side in ('sft','dpo')}}
             for index,(key,question) in enumerate(zip(keys,prompts))]
    write_json(cases_path,cases)
    # Keep the realized allocation seed and model mapping out of reviewer-facing output.
    prepare(cases_path,form_path,mapping_path,seed=secrets.randbits(63))
    protocol.update({'cases_sha256': sha256_file(cases_path), 'unscored_form_sha256': sha256_file(form_path),
                     'private_mapping_sha256': sha256_file(mapping_path), 'review_status': 'unscored'})
    write_json(protocol_path,protocol)
    print(f'Prepared {len(cases)} unseen pairs. Review only {form_path}; do not inspect cases or mapping before locking judgments.')


if __name__=='__main__':
    main()
