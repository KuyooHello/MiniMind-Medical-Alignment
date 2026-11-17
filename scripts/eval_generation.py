import json
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from scripts.eval_ppl import evaluate_ppl

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./model")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

def load_model(ckpt_path: str):
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    return model.to(device=device, dtype=torch.bfloat16).eval()

@torch.no_grad()
def batch_generate(model, prompts: list[str], batch_size: int = 25) -> list[str]:
    all_replies = []
    for start in range(0, len(prompts), batch_size):
        batch_p = prompts[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch_p
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        prompt_len = enc["input_ids"].shape[1]

        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        for i in range(len(batch_p)):
            all_replies.append(tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip())
    return all_replies

if __name__ == "__main__":
    # 1. 验证 DPO+NLL 模型的验证集 PPL（证明加入 NLL 正则后未发生语言退化）
    evaluate_ppl("./out/dpo_nll_768.pth", "./dataset/sft_vertical_mix_val.jsonl")

    # 2. 批量生成 50 条验证集问题的 SFT vs DPO 回复对比
    with open("./dataset/sft_vertical_mix_val.jsonl", "r", encoding="utf-8") as f:
        val_items = [json.loads(line) for line in f][:50]
    prompts = [item["conversations"][0]["content"] for item in val_items]

    sft_model = load_model("./out/full_sft_neftune_768.pth")
    sft_replies = batch_generate(sft_model, prompts)
    del sft_model
    torch.cuda.empty_cache()

    dpo_model = load_model("./out/dpo_nll_768.pth")
    dpo_replies = batch_generate(dpo_model, prompts)

    results = [
        {"id": i + 1, "prompt": p, "sft_reply": s, "dpo_reply": d}
        for i, (p, s, d) in enumerate(zip(prompts, sft_replies, dpo_replies))
    ]
    with open("./out/ablation_eval_cases.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("50 条对比样本已生成至 ./out/ablation_eval_cases.json")