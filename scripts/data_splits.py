"""Shared prompt grouping, split verification, and dataset provenance helpers."""

import hashlib
import json
import random
import re
import unicodedata
from itertools import combinations
from pathlib import Path


DEFAULT_SPLIT_DIR = "dataset/medical"
DEFAULT_DPO_PATH = "dataset/dpo_medical.jsonl"
SPLIT_NAMES = ("train", "val", "test")


def normalize_prompt(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^(?:问|患者|User)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def prompt_key(conversations):
    for message in conversations:
        if message.get("role") == "user":
            text = normalize_prompt(message["content"])
            if not text:
                raise ValueError("用户问题不能为空")
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
    raise ValueError("对话中没有 user 消息")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_outputs_available(paths, overwrite=False):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"文件已存在，使用新路径或显式传入 --overwrite: {existing}")


def split_by_prompt(samples, seed=42, val_ratio=0.05, test_ratio=0.05):
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("val_ratio、test_ratio 必须大于 0，且总和小于 1")
    groups = {}
    for sample in samples:
        groups.setdefault(prompt_key(sample["conversations"]), []).append(sample)
    if len(groups) < 3:
        raise ValueError("至少需要 3 个不同的问题组才能划分 train/val/test")
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio))
    n_test = max(1, int(len(keys) * test_ratio))
    if n_val + n_test >= len(keys):
        raise ValueError("数据太少，当前比例不能生成非空训练集")
    key_splits = {
        "val": keys[:n_val],
        "test": keys[n_val:n_val + n_test],
        "train": keys[n_val + n_test:],
    }
    splits = {}
    for name in SPLIT_NAMES:
        rows = [sample for key in key_splits[name] for sample in sorted(
            groups[key], key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False)
        )]
        rng.shuffle(rows)
        splits[name] = rows
    validate_splits(splits)
    return splits


def validate_splits(splits, dpo_rows=None):
    keys = {}
    for name in SPLIT_NAMES:
        if not splits[name]:
            raise ValueError(f"{name} 数据集为空")
        keys[name] = {prompt_key(row["conversations"]) for row in splits[name]}
    for left, right in combinations(SPLIT_NAMES, 2):
        overlap = keys[left] & keys[right]
        if overlap:
            raise ValueError(f"数据泄漏: {left}/{right} 有 {len(overlap)} 个重复问题组")
    if dpo_rows is not None:
        if not dpo_rows:
            raise ValueError("DPO 数据集为空")
        dpo_keys = set()
        for row in dpo_rows:
            key = prompt_key(row["chosen"])
            if key != prompt_key(row["rejected"]):
                raise ValueError("DPO chosen/rejected 的问题不一致")
            if row.get("prompt_key", key) != key:
                raise ValueError("DPO prompt_key 与问题内容不一致")
            dpo_keys.add(key)
        invalid = dpo_keys - keys["train"]
        if invalid:
            raise ValueError(f"数据泄漏: DPO 有 {len(invalid)} 个问题不属于训练集")
        keys["dpo"] = dpo_keys
    return {name: len(value) for name, value in keys.items()}


def load_verified_splits(split_dir):
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("不支持的数据划分 manifest 版本")
    splits = {}
    for name in SPLIT_NAMES:
        path = split_dir / f"{name}.jsonl"
        expected = manifest["splits"][name]
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"{path} 内容已变化，与 manifest 不一致；请重新构建并检查划分")
        splits[name] = read_jsonl(path)
        if len(splits[name]) != expected["rows"]:
            raise ValueError(f"{name} 样本数量与 manifest 不一致")
    validate_splits(splits)
    return splits, manifest
