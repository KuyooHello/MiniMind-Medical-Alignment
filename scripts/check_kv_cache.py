"""Compare full-prefix inference with KV-cached inference on fixed input tokens."""

import argparse
import math
from pathlib import Path

import torch

from scripts.data_splits import sha256_file, write_json
from scripts.generation_utils import load_model, load_tokenizer


@torch.inference_mode()
def check_forward_equivalence(model, input_ids, chunk_sizes, attention_mask=None,
                              atol=1e-4, rtol=1e-5):
    """Teacher-force identical tokens; compare every valid token's entire logit vector."""
    if model.training:
        raise ValueError("KV Cache 一致性检查要求 model.eval()")
    if not chunk_sizes or any(size < 1 for size in chunk_sizes) or sum(chunk_sizes) != input_ids.shape[1]:
        raise ValueError("chunk_sizes 必须为正且覆盖完整输入")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask 必须覆盖完整输入")
    full = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    if not torch.isfinite(full).all():
        raise AssertionError("完整前向产生非有限 logits")
    past = None
    start = 0
    max_absolute_error = 0.0
    compared_tokens = 0
    num_kv_heads = model.config.num_key_value_heads or model.config.num_attention_heads
    for size in chunk_sizes:
        end = start + size
        old_past = past
        output = model(input_ids[:, start:end], past_key_values=past, use_cache=True,
                       attention_mask=attention_mask[:, :end] if attention_mask is not None else None)
        cached = output.logits
        expected = full[:, start:end]
        valid = (attention_mask[:, start:end].bool() if attention_mask is not None
                 else torch.ones(input_ids.shape[0], size, dtype=torch.bool, device=input_ids.device))
        if valid.any():
            actual_values, expected_values = cached[valid], expected[valid]
            if not torch.isfinite(actual_values).all():
                raise AssertionError(f"缓存前向产生非有限 logits: tokens {start}:{end}")
            try:
                torch.testing.assert_close(actual_values, expected_values, atol=atol, rtol=rtol)
            except AssertionError as error:
                raise AssertionError(f"缓存与完整前向不一致: tokens {start}:{end}\n{error}") from error
            max_absolute_error = max(max_absolute_error,
                                     (actual_values.double()-expected_values.double()).abs().max().item())
            compared_tokens += int(valid.sum())
        past = output.past_key_values
        if past is None or len(past) != model.config.num_hidden_layers:
            raise AssertionError("缓存层数不正确")
        expected_shape = (input_ids.shape[0], end, num_kv_heads, model.config.head_dim)
        for layer_index, pair in enumerate(past):
            for tensor_index, tensor in enumerate(pair):
                if tuple(tensor.shape) != expected_shape:
                    raise AssertionError(f"缓存形状不正确: layer={layer_index}, shape={tuple(tensor.shape)}")
                if old_past is not None:
                    torch.testing.assert_close(tensor[:, :start], old_past[layer_index][tensor_index],
                                               atol=0, rtol=0, msg="续写改变了已有缓存内容")
        start = end
    if not compared_tokens:
        raise ValueError("输入没有有效 token")
    return {"chunk_sizes": list(chunk_sizes), "compared_tokens": compared_tokens,
            "max_absolute_logit_error": max_absolute_error, "cache_shape_checked": True}


@torch.inference_mode()
def check_greedy_equivalence(model, input_ids, attention_mask=None, max_new_tokens=6):
    if model.training:
        raise ValueError("KV Cache 一致性检查要求 model.eval()")
    settings = dict(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=1.0, top_p=1.0, top_k=0, eos_token_id=None)
    cached = model.generate(**settings, use_cache=True)
    uncached = model.generate(**settings, use_cache=False)
    if not torch.equal(cached, uncached):
        raise AssertionError("开启与关闭缓存的贪心生成 token 不一致")
    if cached.shape[1] != input_ids.shape[1] + max_new_tokens:
        raise AssertionError("贪心生成长度不正确")
    return {"max_new_tokens": max_new_tokens, "generated_tokens_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", help="可重复指定；默认检查选定的 SFT 和 DPO 权重")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float64", "float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", default="out/kv_cache_consistency.json")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须大于 0")
    # Batched and single-token matmuls round differently; record the measured error too.
    defaults = {"float64": (1e-8, 1e-8), "float32": (1e-4, 1e-5),
                "bfloat16": (0.1, 0.02), "float16": (0.02, 0.005)}
    atol = defaults[args.dtype][0] if args.atol is None else args.atol
    rtol = defaults[args.dtype][1] if args.rtol is None else args.rtol
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        parser.error("数值容差必须为有限非负数")
    torch.set_num_threads(args.threads)
    tokenizer = load_tokenizer()
    messages = ["你好", "请用一句话解释什么是监督学习。", "请列出三个常见的编程语言名称。"]
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                          tokenize=False, add_generation_prompt=True)
             for message in messages]
    encoded = tokenizer(texts, padding=True, return_tensors="pt").to(args.device)
    generator = torch.Generator().manual_seed(731)
    suffix = torch.randint(3, len(tokenizer), (len(messages), 6), generator=generator).to(args.device)
    ids = torch.cat([encoded["input_ids"], suffix], dim=1)
    mask = torch.cat([encoded["attention_mask"], torch.ones_like(suffix)], dim=1)
    width = encoded["input_ids"].shape[1]
    results = []
    for checkpoint in args.checkpoint or ["out/medical_model_768.pth", "out/medical_dpo_768.pth"]:
        model = load_model(checkpoint, args.device, dtype=getattr(torch, args.dtype))
        row = {"checkpoint": checkpoint, "checkpoint_sha256": sha256_file(checkpoint),
               "single_token_continuation": check_forward_equivalence(
                   model, ids, [width]+[1]*6, mask, atol, rtol),
               "chunked_continuation": check_forward_equivalence(
                   model, ids, [width, 2, 4], mask, atol, rtol),
               "greedy_generation": check_greedy_equivalence(
                   model, encoded["input_ids"], encoded["attention_mask"])}
        results.append(row)
        print(f"PASS {Path(checkpoint).name}: "
              f"tokenwise={row['single_token_continuation']['max_absolute_logit_error']:.3g}, "
              f"chunked={row['chunked_continuation']['max_absolute_logit_error']:.3g}; greedy tokens equal",
              flush=True)
        del model
    write_json(args.output, {"device": args.device, "dtype": args.dtype, "atol": atol, "rtol": rtol,
                             "torch_version": torch.__version__, "batch_size": len(messages),
                             "padding_side": tokenizer.padding_side,
                             "precision_note": "Model RMSNorm and attention softmax retain FP32 intermediates.",
                             "input_policy": "Fixed prompts and teacher-forced token IDs; EOS stopping disabled only for fixed-length greedy comparison.",
                             "passed": True, "results": results})
    print(f"KV Cache 一致性记录 -> {args.output}")


if __name__ == "__main__":
    main()
