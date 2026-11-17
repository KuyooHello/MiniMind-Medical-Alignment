import os
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

@torch.no_grad()
def evaluate_ppl(model_ckpt_path: str, val_jsonl_path: str, device="cuda"):
    if not os.path.exists(model_ckpt_path):
        print(f"[跳过] 未找到权重文件: {model_ckpt_path}")
        return None

    tokenizer = AutoTokenizer.from_pretrained("./model")
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device), strict=False)
    model.to(device).eval()

    total_loss = 0.0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=tokenizer.pad_token_id)

    with open(val_jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f][:300]

    for item in tqdm(lines, desc=os.path.basename(model_ckpt_path)):
        text = tokenizer.apply_chat_template(item["conversations"], tokenize=False)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_ids = enc["input_ids"]
        logits = model(input_ids).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        valid_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
        total_loss += loss.item()
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"权重: {model_ckpt_path} | Val Loss: {avg_loss:.4f} | Val PPL: {ppl:.2f}")
    return ppl

if __name__ == "__main__":
    val_file = "./dataset/sft_vertical_mix_val.jsonl"
    ppl_pre = evaluate_ppl("./out/pretrain_768.pth", val_file)
    ppl_base = evaluate_ppl("./out/full_sft_base_768.pth", val_file)
    ppl_nef = evaluate_ppl("./out/full_sft_neftune_768.pth", val_file)

    if ppl_pre and ppl_nef:
        print(f"\n[评测结论 1] 垂直领域验证集 PPL 较 Pretrain 基线下降: {(ppl_pre - ppl_nef) / ppl_pre * 100:.1f}%")
    if ppl_base and ppl_nef:
        print(f"[评测结论 2] 引入 NEFTune 后验证集 PPL 较原生 SFT 进一步降低: {ppl_base - ppl_nef:.2f} 个点")