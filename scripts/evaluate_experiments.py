"""Evaluate the fixed matrix, choose on validation, then report the sealed test."""

import argparse
import json
import shutil
import statistics
from pathlib import Path

import torch

from scripts.data_splits import load_verified_splits, sha256_file, write_json
from scripts.eval_ppl import evaluate_ppl
from scripts.generation_utils import load_tokenizer


def matrix(config):
    runs = [{"id": "pretrain", "kind": "pretrain", "seed": None, "checkpoint": "out/pretrain_768.pth"}]
    for seed in config["sft"]["seeds"]:
        for kind in ("sft_base", "sft_neftune"):
            name = f"{kind}_seed{seed}"
            runs.append({"id": name, "kind": kind, "seed": seed,
                         "checkpoint": f"out/experiments/{name}_768.pth"})
    for variant in config["dpo"]["variants"]:
        name = variant["name"]
        runs.append({"id": name, "kind": "dpo", "seed": config["dpo"]["settings"]["seed"],
                     "checkpoint": f"out/experiments/{name}_768.pth", **variant["overrides"]})
    return runs


def aggregate(results):
    groups = {}
    for kind in ("sft_base", "sft_neftune"):
        values = [row for row in results if row["kind"] == kind]
        groups[kind] = {"seeds": [row["seed"] for row in values], "n": len(values)}
        for metric in ("loss", "ppl", "response_loss", "response_ppl"):
            observed = [row[metric] for row in values]
            groups[kind][metric] = {"mean": statistics.mean(observed), "sample_std": statistics.stdev(observed)}
    by_id = {row["id"]: row for row in results}
    paired = []
    for seed in groups["sft_base"]["seeds"]:
        base = by_id[f"sft_base_seed{seed}"]
        noisy = by_id[f"sft_neftune_seed{seed}"]
        paired.append({"seed": seed,
                       "ppl_reduction_percent": (base["ppl"] - noisy["ppl"]) / base["ppl"] * 100,
                       "response_ppl_reduction_percent": (base["response_ppl"] - noisy["response_ppl"]) / base["response_ppl"] * 100})
    return {"sft_multiseed": groups, "paired_neftune_changes": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["val", "test"], required=True)
    parser.add_argument("--config", default="configs/experiments.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    directory = Path(config["protocol"]["training_split"]).parent
    splits, manifest = load_verified_splits(directory)
    count = config["protocol"]["validation_samples" if args.phase == "val" else "final_test_samples"]
    rows = splits[args.phase][:count]
    output = Path(f"out/experiments/{'validation' if args.phase == 'val' else 'test'}.json")
    selection_path = Path("out/experiments/selection.json")
    fingerprints = {"config_sha256": sha256_file(args.config),
                    "manifest_sha256": sha256_file(directory / "manifest.json"),
                    "data_sha256": sha256_file(directory / f"{args.phase}.jsonl"),
                    "evaluation_implementation_sha256": sha256_file("scripts/eval_ppl.py")}
    if args.phase == "test":
        selection = json.loads(selection_path.read_text())
        if (selection["config_sha256"] != fingerprints["config_sha256"]
                or selection["manifest_sha256"] != fingerprints["manifest_sha256"]
                or selection.get("evaluation_implementation_sha256") != fingerprints["evaluation_implementation_sha256"]
                or selection["validation_results_sha256"] != sha256_file("out/experiments/validation.json")):
            raise ValueError("验证集选择记录与当前实验不一致，禁止直接评测测试集")
    tokenizer = load_tokenizer()
    results = []
    for run in matrix(config):
        fingerprint = sha256_file(run["checkpoint"])
        if output.exists():
            previous = json.loads(output.read_text())
            if all(previous.get(key) == value for key, value in fingerprints.items()):
                cached = next((row for row in previous["results"] if row["id"] == run["id"]
                               and row["checkpoint_sha256"] == fingerprint), None)
                if cached:
                    results.append(cached)
                    print(f"CACHED {args.phase} {run['id']}", flush=True)
                    continue
        print(f"EVALUATE {args.phase} {run['id']}", flush=True)
        metrics = evaluate_ppl(run["checkpoint"], rows, tokenizer, "cuda", batch_size=args.batch_size)
        results.append({**run, "checkpoint_sha256": fingerprint, **metrics})
        write_json(output, {"phase": args.phase, "samples": len(rows),
                            "metric_scope": "Both full-conversation and assistant-response token NLL/PPL",
                            "batch_size": args.batch_size, **fingerprints, "results": results})
        print(f"RESULT {run['id']}: full={metrics['ppl']:.4f}; response={metrics['response_ppl']:.4f}", flush=True)
        torch.cuda.empty_cache()
    payload = {"phase": args.phase, "samples": len(rows), "batch_size": args.batch_size,
               "metric_scope": "Both full-conversation and assistant-response token NLL/PPL",
               **fingerprints, "results": results, **aggregate(results)}
    write_json(output, payload)
    if args.phase == "val":
        eligible = [row for row in results if row["kind"] == "dpo" or
                    (row["kind"] in {"sft_base", "sft_neftune"} and row["seed"] == 42)]
        winner = min(eligible, key=lambda row: row["response_loss"])
        best_dpo = min((row for row in results if row["kind"] == "dpo"), key=lambda row: row["response_loss"])
        selection = {**fingerprints, "selection_metric": "response_loss", "selection_split": "val",
                     "validation_results_sha256": sha256_file(output),
                     "eligible_models": [row["id"] for row in eligible],
                     "selected": winner["id"], "selected_checkpoint": winner["checkpoint"],
                     "best_dpo": best_dpo["id"], "best_dpo_checkpoint": best_dpo["checkpoint"],
                     "fixed_representative_seed": 42}
        write_json(selection_path, selection)
        write_json("results/validation.json", payload)
        write_json("results/selection.json", selection)
        shutil.copy2(winner["checkpoint"], "out/medical_model_768.pth")
        shutil.copy2(best_dpo["checkpoint"], "out/medical_dpo_768.pth")
        shutil.copy2(Path(winner["checkpoint"]).with_suffix(".training.json"), "out/medical_model_768.training.json")
        shutil.copy2(Path(best_dpo["checkpoint"]).with_suffix(".training.json"), "out/medical_dpo_768.training.json")
        print(f"VALIDATION SELECTED {winner['id']}; best DPO={best_dpo['id']}", flush=True)
    else:
        payload["selection"] = selection
        payload["selection_unchanged_after_test"] = True
        write_json(output, payload)
        write_json("results/experiments.json", payload)
        print("Final test report -> results/experiments.json", flush=True)


if __name__ == "__main__":
    main()
