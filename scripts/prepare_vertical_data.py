import os
import re
import glob
import json
import hashlib
import random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
MAX_LEN = 512

def get_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

def find_file(root_dir: str, pattern: str):
    matches = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
    return matches[0] if matches else None

def parse_conversations(row: dict):
    # 1. 优先读取 conversations 或 HuatuoGPT 的 data 字段
    raw = row.get("conversations") or row.get("data")
    if isinstance(raw, list) and len(raw) >= 2:
        # 情况 A：已经是 [{"role": "user", ...}, {"role": "assistant", ...}]
        if isinstance(raw[0], dict) and "content" in raw[0]:
            return raw
        # 情况 B：HuatuoGPT 格式 ["问：...", "答：..."]（支持单轮及多轮偶数条对话）
        if isinstance(raw[0], str):
            convs = []
            for i in range(0, len(raw) - 1, 2):
                u = re.sub(r"^(问|患者|User)[：:]\s*", "", raw[i]).strip()
                a = re.sub(r"^(答|医生|Assistant)[：:]\s*", "", raw[i + 1]).strip()
                if u and a:
                    convs.append({"role": "user", "content": u})
                    convs.append({"role": "assistant", "content": a})
            return convs

    # 2. 兼容 Alpaca 格式 (instruction + input + output)
    inst = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", "")).strip()
    out = str(row.get("output", "")).strip()
    if inst and out:
        user_text = f"{inst}\n{inp}".strip() if inp else inst
        return [{"role": "user", "content": user_text}, {"role": "assistant", "content": out}]

    return []

def build_sft_dataset(raw_root: str, output_path: str, domain_limit: int = 25000):
    seen_hashes = set()
    domain_samples = []

    all_jsonl = [
        p for p in glob.glob(os.path.join(raw_root, "**", "*.jsonl"), recursive=True)
        if "sft_mini_512" not in os.path.basename(p)
    ]
    if not all_jsonl:
        raise FileNotFoundError(f"未在 {raw_root} 下找到垂直领域 .jsonl 文件")

    print(f"正在离线读取垂直数据集: {all_jsonl[0]}")
    with open(all_jsonl[0], "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            convs = parse_conversations(row)
            if len(convs) < 2:
                continue

            h = get_text_hash(convs[0]["content"] + convs[1]["content"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            formatted_text = tokenizer.apply_chat_template(convs, tokenize=False)
            token_ids = tokenizer.encode(formatted_text)
            if 32 <= len(token_ids) <= MAX_LEN:
                domain_samples.append({"conversations": convs})
                if len(domain_samples) % 5000 == 0:
                    print(f"已收集有效样本: {len(domain_samples)} / {domain_limit}")

            if len(domain_samples) >= domain_limit:
                break

    if not domain_samples:
        raise ValueError("未能提取到有效样本，请检查原始 JSONL 的字段结构。")

    # 按 7:3 混入通用 SFT 数据（若存在）
    general_path = find_file(raw_root, "sft_mini_512.jsonl") or "./dataset/sft_mini_512.jsonl"
    general_samples = []
    if os.path.exists(general_path):
        general_limit = int(len(domain_samples) * (3 / 7))
        with open(general_path, "r", encoding="utf-8") as f:
            for line in f:
                general_samples.append(json.loads(line))
                if len(general_samples) >= general_limit:
                    break
        print(f"已混入通用 SFT 数据: {len(general_samples)} 条")
    else:
        print("未检测到 sft_mini_512.jsonl，将仅使用垂直领域数据构建训练集。")

    merged = domain_samples + general_samples
    random.seed(42)
    random.shuffle(merged)

    split_idx = int(len(merged) * 0.95)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in merged[:split_idx]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val_path = output_path.replace(".jsonl", "_val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in merged[split_idx:]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"首条样本预览: {merged[0]['conversations'][0]['content'][:40]}...")
    print(f"构建完成 -> 训练集: {output_path} ({split_idx} 条) | 验证集: {val_path} ({len(merged) - split_idx} 条)")

if __name__ == "__main__":
    build_sft_dataset(
        raw_root="./dataset/raw",
        output_path="./dataset/sft_vertical_mix.jsonl"
    )