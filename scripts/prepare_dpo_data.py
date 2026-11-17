import json
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
MAX_LEN = 512

def build_dpo_dataset(hf_name: str, output_path: str, limit: int = 6000):
    ds = load_dataset(hf_name, split="train")
    valid_pairs = []

    for row in ds:
        # 兼容常见 DPO 数据集字段（以 jondurbin/py-dpo-v0.1 或同类指令偏好集为例）
        prompt = row.get("prompt") or row.get("instruction") or row.get("question")
        chosen_resp = row.get("chosen")
        rejected_resp = row.get("rejected")

        # 若 chosen 本身已是对话列表则直接使用，否则封装为标准消息格式
        if isinstance(chosen_resp, list) and isinstance(rejected_resp, list):
            chosen_conv, rejected_conv = chosen_resp, rejected_resp
        elif prompt and isinstance(chosen_resp, str) and isinstance(rejected_resp, str):
            chosen_conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_resp}]
            rejected_conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_resp}]
        else:
            continue

        # 校验两条序列在 tokenize 后均不超过 MAX_LEN，防止截断破坏偏好信号
        c_len = len(tokenizer.encode(tokenizer.apply_chat_template(chosen_conv, tokenize=False)))
        r_len = len(tokenizer.encode(tokenizer.apply_chat_template(rejected_conv, tokenize=False)))

        if 20 <= c_len <= MAX_LEN and 20 <= r_len <= MAX_LEN:
            valid_pairs.append({"chosen": chosen_conv, "rejected": rejected_conv})

        if len(valid_pairs) >= limit:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        for item in valid_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"DPO 数据集构建完成，共 {len(valid_pairs)} 对有效偏好样本 -> {output_path}")

if __name__ == "__main__":
    # 若做医疗方向可替换为 FreedomIntelligence/HuatuoGPT-rlhf-data，数学方向可用 argilla/distilabel-math-preference-dpo
    build_dpo_dataset("jondurbin/py-dpo-v0.1", "./dataset/dpo_vertical.jsonl")