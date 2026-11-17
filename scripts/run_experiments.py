"""Run the predeclared experiment matrix and record every command and checkpoint."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.data_splits import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/experiments"


def train(name, script, settings, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / f"{name}_768.pth"
    metadata = OUT / f"{name}_768.training.json"
    if checkpoint.exists():
        if not metadata.exists() or not json.loads(metadata.read_text()).get("completed"):
            raise RuntimeError(f"未完成的训练输出已存在: {checkpoint}")
        saved = json.loads(metadata.read_text())
        requested = {**settings, **(extra or {})}
        if any(saved["settings"].get(key) != value for key, value in requested.items()):
            raise RuntimeError(f"已有训练的参数与当前计划不一致: {name}")
        if saved.get("checkpoint_sha256") and saved["checkpoint_sha256"] != sha256_file(checkpoint):
            raise RuntimeError(f"已有权重哈希发生变化: {name}")
        expected_data = sha256_file(ROOT / requested["data_path"].removeprefix("../"))
        if saved["data_sha256"] != expected_data:
            raise RuntimeError(f"已完成训练的数据发生变化: {name}")
        print(f"SKIP completed {name}", flush=True)
        return
    command = [sys.executable, "-u", script, "--save_dir", "../out/experiments",
               "--save_weight", name, "--hidden_size", "768", "--num_hidden_layers", "8",
               "--max_seq_len", "512", "--num_workers", "8", "--log_interval", "100",
               "--save_interval", "100000"]
    for key, value in {**settings, **(extra or {})}.items():
        command.extend([f"--{key}", str(value)])
    log_path = OUT / f"{name}.log"
    start = time.time()
    write_json(OUT / f"{name}.command.json", {"cwd": "trainer", "command": command,
                                            "start_unix": start})
    print(f"START {name}; log={log_path}", flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                       HF_DATASETS_CACHE=str(ROOT / "out/datasets_cache"))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT / "trainer", env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = log_path.read_text(errors="replace")[-5000:]
        raise RuntimeError(f"{name} failed ({process.returncode}):\n{tail}")
    finished = json.loads(metadata.read_text())
    if not finished.get("completed"):
        raise RuntimeError(f"{name} 缺少训练完成记录")
    finished.update({"elapsed_seconds": time.time() - start, "checkpoint_sha256": sha256_file(checkpoint)})
    write_json(metadata, finished)
    print(f"DONE {name}; elapsed={finished['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "dpo"], required=True)
    parser.add_argument("--config", default="configs/experiments.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.stage == "sft":
        for seed in config["sft"]["seeds"]:
            for variant, alpha in [("sft_base", 0), ("sft_neftune", 5)]:
                train(f"{variant}_seed{seed}", "train_full_sft.py",
                      {**config["sft"]["settings"], "seed": seed, "neftune_alpha": alpha})
        for variant in ("sft_base", "sft_neftune"):
            checkpoint = OUT / f"{variant}_seed42_768.pth"
            shutil.copy2(checkpoint, ROOT / f"out/medical_{variant}_768.pth")
            shutil.copy2(checkpoint.with_suffix(".training.json"),
                         ROOT / f"out/medical_{variant}_768.training.json")
    else:
        for variant in config["dpo"]["variants"]:
            train(variant["name"], "train_dpo.py",
                  config["dpo"]["settings"], variant["overrides"])


if __name__ == "__main__":
    main()
