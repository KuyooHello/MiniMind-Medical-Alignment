"""Export verified local experiment records without publishing full datasets or weights."""

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

from scripts.data_splits import (
    load_verified_splits, prompt_key, read_jsonl, sha256_file,
    validate_splits, write_json,
)


def export(cases_path):
    config = json.loads(Path("configs/experiments.json").read_text())
    directory = Path(config["protocol"]["training_split"]).parent
    splits, manifest = load_verified_splits(directory)
    keys = {name: {prompt_key(row["conversations"]) for row in rows}
            for name, rows in splits.items()}
    if keys["test"] & set(manifest["domain_prompt_keys"]):
        raise ValueError("最终测试集含有核心语料中的问题")
    public_manifest = {key: value for key, value in manifest.items() if key != "domain_prompt_keys"}
    public_manifest.update({
        "original_manifest_sha256": sha256_file(directory / "manifest.json"),
        "pairwise_prompt_intersections": {f"{a}/{b}": len(keys[a] & keys[b])
                                          for a, b in combinations(keys, 2)},
        "fresh_test_core_prompt_intersection": 0,
        "test_prompt_keys": sorted(keys["test"]),
    })
    write_json("results/data_manifest.json", public_manifest)

    report = json.loads(Path("out/experiments/test.json").read_text())
    if (report["config_sha256"] != sha256_file("configs/experiments.json")
            or report["manifest_sha256"] != public_manifest["original_manifest_sha256"]
            or report["data_sha256"] != manifest["splits"]["test"]["sha256"]):
        raise ValueError("测试记录与当前配置或划分不一致")
    selection_path = Path("out/experiments/selection.json")
    selection = json.loads(selection_path.read_text())
    if (selection != report["selection"]
            or selection["validation_results_sha256"] != sha256_file("out/experiments/validation.json")):
        raise ValueError("验证集选择记录发生变化")
    for source, target in [("test", "experiments"), ("validation", "validation"), ("selection", "selection")]:
        shutil.copy2(f"out/experiments/{source}.json", f"results/{target}.json")

    training = []
    for run in report["results"]:
        if run["kind"] == "pretrain":
            continue
        metadata = json.loads(Path(run["checkpoint"]).with_suffix(".training.json").read_text())
        if not metadata["completed"] or sha256_file(run["checkpoint"]) != run["checkpoint_sha256"]:
            raise ValueError(f"训练未完成或权重不一致: {run['id']}")
        data_path = Path(metadata["settings"]["data_path"].removeprefix("../"))
        if sha256_file(data_path) != metadata["data_sha256"]:
            raise ValueError(f"训练数据发生变化: {run['id']}")
        training.append({"id": run["id"], **{key: metadata[key] for key in (
            "completed", "settings", "data_sha256", "initial_checkpoint_sha256", "checkpoint_sha256",
            "source_sha256", "provenance", "completed_epochs", "last_step") if key in metadata}})
    write_json("results/training_runs.json", {"runs": training})

    quality = json.loads(Path("out/preference_quality.json").read_text())
    semantic_path = Path("results/preference_sample_review.json")
    if (quality["semantic_review_sha256"] != sha256_file(semantic_path)
            or quality["input_sha256"] != sha256_file("dataset/dpo_medical.jsonl")
            or quality["output_sha256"] != sha256_file("dataset/dpo_medical_filtered.jsonl")):
        raise ValueError("偏好数据或审阅记录的哈希不一致")
    for name in ("dpo_medical", "dpo_medical_filtered"):
        validate_splits(splits, read_jsonl(f"dataset/{name}.jsonl"))
    quality = {key: value for key, value in quality.items() if key != "pair_decisions"}
    quality["review_method"] = json.loads(semantic_path.read_text())["review_method"]
    write_json("results/preference_quality.json", quality)

    metadata = json.loads(Path(f"{cases_path}.metadata.json").read_text())
    cases = json.loads(Path(cases_path).read_text())
    if metadata["cases_sha256"] != sha256_file(cases_path):
        raise ValueError("生成样例内容发生变化")
    if metadata["data_sha256"] != manifest["splits"]["test"]["sha256"]:
        raise ValueError("生成样例未使用最终测试集")
    if len({row["prompt_key"] for row in cases}) != len(cases):
        raise ValueError("生成样例的问题重复")
    if not {row["prompt_key"] for row in cases} <= keys["test"]:
        raise ValueError("生成样例有测试集以外的问题")
    shutil.copy2(cases_path, "examples/eval_cases_sample.json")
    metadata["published_cases_path"] = "examples/eval_cases_sample.json"
    write_json("results/generation_metadata.json", metadata)
    print(f"Exported {len(training)} training records and {len(cases)} fixed generation examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="out/generation_test.json")
    export(parser.parse_args().cases)
