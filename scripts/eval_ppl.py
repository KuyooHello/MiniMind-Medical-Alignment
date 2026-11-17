"""Evaluate identical held-out examples and save the evaluation provenance."""

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.data_splits import (
    DEFAULT_SPLIT_DIR, ensure_outputs_available, load_verified_splits, prompt_key,
    read_jsonl, sha256_file, validate_splits, write_json,
)
from scripts.generation_utils import load_model, load_tokenizer


def assistant_token_mask(ids, start_ids, end_ids):
    """Match the training response mask without stochastic chat preprocessing."""
    mask = [0] * len(ids)
    index = 0
    while index < len(ids):
        if ids[index:index + len(start_ids)] != start_ids:
            index += 1
            continue
        start = index + len(start_ids)
        end = start
        while end < len(ids) and ids[end:end + len(end_ids)] != end_ids:
            end += 1
        stop = min(end + len(end_ids), len(ids))
        mask[start:stop] = [1] * (stop - start)
        index = stop
    return mask


@torch.inference_mode()
def evaluate_ppl(model_ckpt_path, rows, tokenizer, device, max_length=512,
                 hidden_size=768, num_hidden_layers=8, batch_size=16):
    model = load_model(model_ckpt_path, device, hidden_size, num_hidden_layers, dtype=torch.float32)
    total_loss, total_tokens, response_loss, response_tokens = 0.0, 0, 0.0, 0
    start_ids = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    end_ids = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    for start in tqdm(range(0, len(rows), batch_size), desc=Path(model_ckpt_path).name):
        texts = [tokenizer.apply_chat_template(item["conversations"], tokenize=False)
                 for item in rows[start:start + batch_size]]
        encoded = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        response_mask = torch.tensor([assistant_token_mask(ids, start_ids, end_ids)
                                      for ids in encoded["input_ids"].tolist()], dtype=torch.bool, device=device)[:, 1:]
        encoded = encoded.to(device)
        input_ids = encoded["input_ids"]
        logits = model(input_ids, attention_mask=encoded["attention_mask"]).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            reduction="none", ignore_index=tokenizer.pad_token_id,
        ).view_as(labels)
        # A prediction requires both a real target and a real preceding token.
        # With left padding, the first real token otherwise gets scored from PAD.
        attention = encoded["attention_mask"].bool()
        valid = (labels != tokenizer.pad_token_id) & attention[:, 1:] & attention[:, :-1]
        response_mask &= valid
        total_loss += losses[valid].sum().item()
        total_tokens += valid.sum().item()
        response_loss += losses[response_mask].sum().item()
        response_tokens += response_mask.sum().item()
    if total_tokens == 0:
        raise ValueError("没有可评测的 token")
    average = total_loss / total_tokens
    if not response_tokens:
        raise ValueError("没有有效 assistant 回复 token，请检查模板与最大长度")
    response_average = response_loss / response_tokens
    return {"loss": average, "ppl": math.exp(average), "tokens": total_tokens, "samples": len(rows),
            "response_loss": response_average, "response_ppl": math.exp(response_average),
            "response_tokens": response_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", action="append", help="可重复指定；默认比较预训练、SFT、NEFTune 与选定的 DPO")
    parser.add_argument("--dpo-path", help="可选：同时检查本次使用的 DPO 训练数据")
    parser.add_argument("--num-samples", type=int, default=300, help="0 表示评测整个集合")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tokenizer-path", default="model")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 0 or args.max_length < 2 or args.batch_size < 1:
        parser.error("--num-samples 必须非负，--max-length 至少为 2")
    output = args.output or f"out/ppl_{args.split}.json"
    ensure_outputs_available([output], args.overwrite)
    splits, _ = load_verified_splits(args.split_dir)
    if args.dpo_path:
        validate_splits(splits, read_jsonl(args.dpo_path))
    rows = splits[args.split]
    if args.num_samples:
        rows = rows[:args.num_samples]
    checkpoints = args.checkpoint or [
        "out/pretrain_768.pth", "out/medical_sft_base_768.pth",
        "out/medical_sft_neftune_768.pth", "out/medical_dpo_768.pth",
    ]
    missing = [path for path in checkpoints if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少模型权重，请先完成对应训练: {missing}")
    tokenizer = load_tokenizer(args.tokenizer_path)
    results = []
    for checkpoint in checkpoints:
        metrics = evaluate_ppl(checkpoint, rows, tokenizer, args.device, args.max_length,
                               args.hidden_size, args.num_hidden_layers, args.batch_size)
        results.append({"checkpoint": checkpoint, "checkpoint_sha256": sha256_file(checkpoint), **metrics})
        print(f"{checkpoint} | {args.split} loss: {metrics['loss']:.4f} | PPL: {metrics['ppl']:.2f} | response PPL: {metrics['response_ppl']:.2f}")
    write_json(output, {
        "metric_scope": "Both full-conversation and assistant-response token NLL/PPL",
        "settings": vars(args),
        "manifest_sha256": sha256_file(Path(args.split_dir) / "manifest.json"),
        "data_sha256": sha256_file(Path(args.split_dir) / f"{args.split}.jsonl"),
        "prompt_keys": [prompt_key(row["conversations"]) for row in rows],
        "results": results,
    })
    print(f"评测记录 -> {output}")


if __name__ == "__main__":
    main()
