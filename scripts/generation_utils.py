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
                     max_new_tokens=180, temperature=1.0, top_p=0.92,
                     top_k=50, repetition_penalty=1.0, return_metadata=False):
    if batch_size < 1 or max_new_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("生成参数不合法")
    if top_k < 0 or repetition_penalty <= 0:
        raise ValueError("top_k 必须非负，repetition_penalty 必须为正")
    replies, metadata = [], []
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
            top_k=top_k, repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        for row in outputs:
            generated = row[prompt_length:].tolist()
            reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
            replies.append(reply)
            if return_metadata:
                eos_position = next((index for index, token in enumerate(generated)
                                     if token == tokenizer.eos_token_id), None)
                length = eos_position + 1 if eos_position is not None else len(generated)
                metadata.append({'generated_tokens': length, 'eos_seen': eos_position is not None,
                                 'generation_cap_without_eos': eos_position is None and length == max_new_tokens})
    return (replies, metadata) if return_metadata else replies
