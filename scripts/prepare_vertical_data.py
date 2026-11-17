"""Clean once, then assign every version of a question to the same split."""

import argparse
import glob
import json
import random
import re
from pathlib import Path

from scripts.data_splits import (
    DEFAULT_SPLIT_DIR, SPLIT_NAMES, ensure_outputs_available, prompt_key,
    sha256_file, split_by_prompt, validate_splits, write_json, write_jsonl,
)


def parse_conversations(row):
    raw = row.get("conversations") or row.get("data")
    if isinstance(raw, list) and len(raw) >= 2:
        if all(isinstance(message, dict) for message in raw):
            if all(isinstance(message.get("content"), str) and message["content"].strip()
                   and message.get("role") in {"system", "user", "assistant"} for message in raw):
                return raw
        elif all(isinstance(message, str) for message in raw):
            messages = []
            for index in range(0, len(raw) - 1, 2):
                question = re.sub(r"^(问|患者|User)[：:]\s*", "", raw[index]).strip()
                answer = re.sub(r"^(答|医生|Assistant)[：:]\s*", "", raw[index + 1]).strip()
                if not question or not answer:
                    return []
                messages.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}])
            return messages
    instruction = row.get("instruction")
    answer = row.get("output")
    extra = row.get("input") or ""
    if isinstance(instruction, str) and isinstance(answer, str) and instruction.strip() and answer.strip():
        question = f"{instruction.strip()}\n{extra}".strip() if extra else instruction.strip()
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer.strip()}]
    return []


def collect_samples(paths, tokenizer, limit, seen, min_length, max_length, exclude_keys=None):
    samples = []
    if limit <= 0:
        return samples
    for path in paths:
        print(f"读取数据: {path}", flush=True)
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    conversations = parse_conversations(json.loads(line))
                    if not conversations or conversations[-1]["role"] != "assistant":
                        continue
                    key = prompt_key(conversations)
                    if exclude_keys and key in exclude_keys:
                        continue
                    text = tokenizer.apply_chat_template(conversations, tokenize=False)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"无法解析 {path}:{line_number}: {error}") from error
                size = len(tokenizer.encode(text, add_special_tokens=False))
                if not min_length <= size <= max_length:
                    continue
                identity = json.dumps(conversations, ensure_ascii=False, sort_keys=True)
                if identity in seen:
                    continue
                seen.add(identity)
                samples.append({"conversations": conversations})
                if len(samples) % 5000 == 0:
                    print(f"已收集 {len(samples)} / {limit} 条有效样本", flush=True)
                if len(samples) >= limit:
                    return samples
    return samples


def build_sft_dataset(args):
    from transformers import AutoTokenizer

    output_dir = Path(args.output_dir)
    ensure_outputs_available(
        [output_dir / f"{name}.jsonl" for name in SPLIT_NAMES] + [output_dir / "manifest.json"],
        args.overwrite,
    )
    if args.domain_limit <= 0 or args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("domain-limit 和长度范围不合法")
    if args.fresh_test_size < 0 or args.fresh_test_pool < args.fresh_test_size:
        raise ValueError("fresh-test-size 必须非负，fresh-test-pool 至少与其相等")
    domain_paths = sorted(Path(path) for path in glob.glob(f"{args.raw_root}/**/*.jsonl", recursive=True)
                          if "sft_mini_512" not in Path(path).name)
    if not domain_paths:
        raise FileNotFoundError(f"未在 {args.raw_root} 找到领域 JSONL 数据")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    seen = set()
    domain = collect_samples(domain_paths, tokenizer, args.domain_limit, seen, args.min_length, args.max_length)
    general_path = Path(args.general_path) if args.general_path else None
    if general_path is None:
        candidates = sorted(Path(args.raw_root).rglob("sft_mini_512.jsonl"))
        fallback = Path("dataset/sft_mini_512.jsonl")
        general_path = candidates[0] if candidates else (fallback if fallback.exists() else None)
    general = []
    if general_path:
        general = collect_samples([general_path], tokenizer, int(len(domain) * 3 / 7), seen,
                                  args.min_length, args.max_length)
    else:
        print("未提供通用数据，本次仅使用医疗数据，不标记为 7:3 混合。", flush=True)
    splits = split_by_prompt(domain + general, args.seed, args.val_ratio, args.test_ratio)
    unused_development_rows = 0
    if args.fresh_test_size:
        core_keys = {prompt_key(row["conversations"]) for row in domain + general}
        pool = collect_samples(domain_paths, tokenizer, args.fresh_test_pool, set(),
                               args.min_length, args.max_length, exclude_keys=core_keys)
        unique = {}
        for row in pool:
            unique.setdefault(prompt_key(row["conversations"]), row)
        if len(unique) < args.fresh_test_size:
            raise ValueError("训练语料之外的有效问题不足以封存新测试集")
        unused_development_rows = len(splits["test"])
        splits["test"] = random.Random(args.seed + 10000).sample(list(unique.values()), args.fresh_test_size)
    group_counts = validate_splits(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "grouping": "NFKC + whitespace + role-prefix normalized first user question; exact matches only",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "tokenizer_sha256": sha256_file(Path(args.tokenizer_path) / "tokenizer.json"),
        "domain_rows": len(domain),
        "general_rows": len(general),
        "fresh_test_size": args.fresh_test_size,
        "fresh_test_pool": args.fresh_test_pool if args.fresh_test_size else 0,
        "unused_development_rows": unused_development_rows,
        "domain_prompt_keys": sorted({prompt_key(row["conversations"]) for row in domain}),
        "sources": [{"path": str(path), "sha256": sha256_file(path)} for path in
                    domain_paths + ([general_path] if general_path else [])],
        "splits": {},
    }
    for name, rows in splits.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][name] = {"rows": len(rows), "prompt_groups": group_counts[name],
                                    "sha256": sha256_file(path)}
        print(f"{name}: {len(rows)} 条，{group_counts[name]} 个问题组 -> {path}", flush=True)
    write_json(output_dir / "manifest.json", manifest)
    print("检查通过：train/val/test 的规范化问题交集均为 0。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="dataset/raw")
    parser.add_argument("--output-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--general-path", help="可选的通用 SFT JSONL；未指定时尝试查找 sft_mini_512.jsonl")
    parser.add_argument("--tokenizer-path", default="model")
    parser.add_argument("--domain-limit", type=int, default=25000)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh-test-size", type=int, default=0, help="从核心语料以外封存的新测试问题数；0 保留常规划分")
    parser.add_argument("--fresh-test-pool", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    build_sft_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
