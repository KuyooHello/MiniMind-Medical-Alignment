"""Build weakly labelled DPO pairs exclusively from the verified training split."""

import argparse
import random
from pathlib import Path

import torch

from scripts.data_splits import (
    DEFAULT_DPO_PATH, DEFAULT_SPLIT_DIR, ensure_outputs_available, load_verified_splits,
    prompt_key, sha256_file, validate_splits, write_json, write_jsonl,
)
from scripts.generation_utils import generate_replies, load_model, load_tokenizer, seed_generation


def select_candidates(rows, domain_keys, tokenizer, limit, max_length, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    candidates, seen = [], set()
    for row in shuffled:
        messages = row["conversations"]
        key = prompt_key(messages)
        if key not in domain_keys or key in seen:
            continue
        first_user = next(index for index, message in enumerate(messages) if message["role"] == "user")
        if first_user + 1 >= len(messages) or messages[first_user + 1]["role"] != "assistant":
            continue
        chosen = [messages[first_user], messages[first_user + 1]]
        text = tokenizer.apply_chat_template(chosen, tokenize=False)
        if not 40 <= len(tokenizer.encode(text, add_special_tokens=False)) <= max_length:
            continue
        seen.add(key)
        candidates.append((key, chosen, len(tokenizer.encode(chosen[0]["content"], add_special_tokens=False))))
        if len(candidates) >= limit:
            break
    # Token length, rather than character length, controls the actual padding cost.
    candidates.sort(key=lambda item: item[2])
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--checkpoint", default="out/medical_sft_neftune_768.pth")
    parser.add_argument("--output", default=DEFAULT_DPO_PATH)
    parser.add_argument("--tokenizer-path", default="model")
    parser.add_argument("--limit", type=int, default=2500, help="最多采样的问题数；过滤后有效偏好对可能更少")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须大于 0")
    metadata_path = f"{args.output}.metadata.json"
    ensure_outputs_available([args.output, metadata_path], args.overwrite)
    splits, manifest = load_verified_splits(args.split_dir)
    tokenizer = load_tokenizer(args.tokenizer_path)
    candidates = select_candidates(splits["train"], set(manifest["domain_prompt_keys"]),
                                   tokenizer, args.limit, args.max_length, args.seed)
    if not candidates:
        raise ValueError("训练集中没有满足长度条件的医疗偏好候选问题")
    seed_generation(args.seed)
    model = load_model(args.checkpoint, args.device, args.hidden_size, args.num_hidden_layers)
    prompts = [chosen[0]["content"] for _, chosen, _ in candidates]
    replies = generate_replies(model, tokenizer, prompts, args.device, args.batch_size,
                               args.max_new_tokens, args.temperature, args.top_p)
    pairs = []
    for (key, chosen, _), reply in zip(candidates, replies):
        if len(reply) < 10 or reply == chosen[1]["content"].strip():
            continue
        rejected = [chosen[0], {"role": "assistant", "content": reply}]
        text = tokenizer.apply_chat_template(rejected, tokenize=False)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= args.max_length:
            pairs.append({"prompt_key": key, "chosen": chosen, "rejected": rejected})
    validate_splits(splits, pairs)
    write_jsonl(args.output, pairs)
    write_json(metadata_path, {
        "schema_version": 1,
        "label_source": "weak labels: dataset answer as chosen; SFT sample as rejected; not quality-verified",
        "settings": vars(args),
        "sampled_prompts": len(candidates),
        "pairs": len(pairs),
        "manifest_sha256": sha256_file(Path(args.split_dir) / "manifest.json"),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_sha256": sha256_file(args.output),
    })
    print(f"已生成 {len(pairs)} 对弱标注偏好数据 -> {args.output}")
    print("检查通过：全部问题属于 train，与 val/test 的规范化问题交集为 0。")


if __name__ == "__main__":
    main()
