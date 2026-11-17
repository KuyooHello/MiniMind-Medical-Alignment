import os
import re
import glob
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./model")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

MAX_LEN = 512
BATCH_SIZE = 32

def clean_prefix(text: str) -> str:
    return re.sub(r"^(问|答|患者|医生|User|Assistant)[：:]\s*", "", str(text)).strip()

@torch.no_grad()
def generate_rejected_batch(model, prompts: list[str]) -> list[str]:
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    prompt_len = enc["input_ids"].shape[1]

    outputs = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=180,
        do_sample=True,
        temperature=1.0,
        top_p=0.92,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    replies = []
    for i in range(len(prompts)):
        gen_ids = outputs[i][prompt_len:]
        replies.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return replies

def main():
    ckpt = "./out/full_sft_neftune_768.pth"
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    # 启用 bfloat16 触发 Tensor Core 推理加速
    model.to(device=device, dtype=torch.bfloat16).eval()

    files = glob.glob("./dataset/raw/**/*HuatuoGPT*.jsonl", recursive=True)
    candidates = []
    with open(files[0], "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < 30000:
                continue
            row = json.loads(line)
            raw = row.get("data") or []
            if len(raw) >= 2 and isinstance(raw[0], str):
                q, a = clean_prefix(raw[0]), clean_prefix(raw[1])
                if q and a:
                    chosen_conv = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                    c_len = len(tokenizer.encode(tokenizer.apply_chat_template(chosen_conv, tokenize=False)))
                    if 40 <= c_len <= MAX_LEN:
                        candidates.append((q, chosen_conv, len(q)))
            if len(candidates) >= 2500:
                break

    # 按问题长度排序后再分批，最小化 left-padding 开销并保持 RoPE 位置精确
    candidates.sort(key=lambda x: x[2])

    dpo_pairs = []
    for start in tqdm(range(0, len(candidates), BATCH_SIZE), desc="Batch Generating DPO pairs"):
        batch = candidates[start:start + BATCH_SIZE]
        prompts = [item[0] for item in batch]
        rej_texts = generate_rejected_batch(model, prompts)

        for (q, chosen_conv, _), rej_text in zip(batch, rej_texts):
            if len(rej_text) < 10 or rej_text == chosen_conv[1]["content"]:
                continue
            rejected_conv = [{"role": "user", "content": q}, {"role": "assistant", "content": rej_text}]
            r_len = len(tokenizer.encode(tokenizer.apply_chat_template(rejected_conv, tokenize=False)))
            if r_len <= MAX_LEN:
                dpo_pairs.append({"chosen": chosen_conv, "rejected": rejected_conv})

    with open("./dataset/dpo_vertical.jsonl", "w", encoding="utf-8") as out:
        for item in dpo_pairs:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"On-Policy 医疗 DPO 数据集构建完成，共 {len(dpo_pairs)} 对 -> ./dataset/dpo_vertical.jsonl")

if __name__ == "__main__":
    main()
