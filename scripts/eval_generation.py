"""Generate paired replies on a verified held-out split; this does not assign wins."""

import argparse
import json
import random
from pathlib import Path

import torch

from scripts.data_splits import (
    DEFAULT_DPO_PATH, DEFAULT_SPLIT_DIR, ensure_outputs_available, load_verified_splits,
    prompt_key, read_jsonl, sha256_file, validate_splits, write_json,
)
from scripts.generation_utils import generate_replies, load_model, load_tokenizer, seed_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--dpo-path", help="默认读取权重旁训练记录中的实际数据路径")
    parser.add_argument("--sft-checkpoint", default="out/medical_sft_neftune_768.pth")
    parser.add_argument("--dpo-checkpoint", default="out/medical_dpo_768.pth")
    parser.add_argument("--tokenizer-path", default="model")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples 必须大于 0")
    output = args.output or f"out/generation_{args.split}.json"
    metadata_path = f"{output}.metadata.json"
    ensure_outputs_available([output, metadata_path], args.overwrite)
    splits, _ = load_verified_splits(args.split_dir)
    training_path = Path(args.dpo_checkpoint).with_suffix(".training.json")
    training = json.loads(training_path.read_text()) if training_path.exists() else None
    if args.dpo_path is None:
        args.dpo_path = (training["settings"]["data_path"].removeprefix("../")
                         if training else DEFAULT_DPO_PATH)
    if training and training["data_sha256"] != sha256_file(args.dpo_path):
        raise ValueError("DPO 数据与权重训练记录中的哈希不一致")
    validate_splits(splits, read_jsonl(args.dpo_path))
    unique_rows = {}
    for row in splits[args.split]:
        unique_rows.setdefault(prompt_key(row["conversations"]), row)
    selected = random.Random(args.seed).sample(list(unique_rows.values()), min(args.num_samples, len(unique_rows)))
    prompts = [next(message["content"] for message in row["conversations"] if message["role"] == "user")
               for row in selected]
    tokenizer = load_tokenizer(args.tokenizer_path)
    replies = {}
    for name, checkpoint in [("sft", args.sft_checkpoint), ("dpo", args.dpo_checkpoint)]:
        seed_generation(args.seed)
        model = load_model(checkpoint, args.device, args.hidden_size, args.num_hidden_layers)
        replies[name] = generate_replies(model, tokenizer, prompts, args.device, args.batch_size,
                                         args.max_new_tokens, args.temperature, args.top_p)
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    cases = [
        {"id": index + 1, "prompt_key": prompt_key(row["conversations"]), "prompt": question,
         "sft_reply": sft, "dpo_reply": dpo}
        for index, (row, question, sft, dpo) in enumerate(zip(selected, prompts, replies["sft"], replies["dpo"]))
    ]
    write_json(output, cases)
    write_json(metadata_path, {
        "settings": vars(args),
        "samples": len(cases),
        "manifest_sha256": sha256_file(Path(args.split_dir) / "manifest.json"),
        "data_sha256": sha256_file(Path(args.split_dir) / f"{args.split}.jsonl"),
        "dpo_data_sha256": sha256_file(args.dpo_path),
        "sft_checkpoint_sha256": sha256_file(args.sft_checkpoint),
        "dpo_checkpoint_sha256": sha256_file(args.dpo_checkpoint),
        "cases_sha256": sha256_file(output),
        "prompt_keys": [case["prompt_key"] for case in cases],
        "judging": "not performed; randomize A/B order and save item-level judgments before reporting win rate",
    })
    print(f"{len(cases)} 条生成对比 -> {output}；此脚本不自动产生盲测胜率。")


if __name__ == "__main__":
    main()
