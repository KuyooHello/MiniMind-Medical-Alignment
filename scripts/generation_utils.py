"""Shared inference settings for preference sampling and paired evaluation."""

import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def seed_generation(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(path="model"):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(checkpoint, device, hidden_size=768, num_hidden_layers=8, dtype=None):
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=False)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if dtype is None:
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if str(device).startswith("cuda") else torch.float32
    return model.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def generate_replies(model, tokenizer, prompts, device, batch_size=32,
                     max_new_tokens=180, temperature=1.0, top_p=0.92):
    if batch_size < 1 or max_new_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("生成参数不合法")
    replies = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True)
            for prompt in prompts[start:start + batch_size]
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        prompt_length = encoded["input_ids"].shape[1]
        outputs = model.generate(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        replies.extend(tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip() for row in outputs)
    return replies
