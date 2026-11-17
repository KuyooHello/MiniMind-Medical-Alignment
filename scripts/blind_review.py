"""Prepare a blinded paired review, then summarize completed item-level judgments."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from scripts.data_splits import ensure_outputs_available, sha256_file, write_json


DEFAULT_FORM = "out/preference_review.csv"
DEFAULT_MAPPING = "out/preference_review.mapping.json"
FIELDS = ("review_id", "prompt", "answer_a", "answer_b", "winner", "reason")


def content_digest(row):
    content = {key: str(row[key]).replace("\r\n", "\n").replace("\r", "\n")
               for key in ("review_id", "prompt", "answer_a", "answer_b")}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def prepare(cases_path, form_path, mapping_path, seed=42, overwrite=False):
    ensure_outputs_available([form_path, mapping_path], overwrite)
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("生成对比为空，或样本 id 重复")
    if len({case["prompt_key"] for case in cases}) != len(cases):
        raise ValueError("生成对比包含重复问题")
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    a_models = ["dpo"] * (len(cases) // 2) + ["sft"] * (len(cases) - len(cases) // 2)
    rng.shuffle(a_models)
    rows, mappings = [], []
    for index, (case, a_model) in enumerate(zip(shuffled, a_models), 1):
        b_model = "sft" if a_model == "dpo" else "dpo"
        row = {"review_id": str(index), "prompt": case["prompt"],
               "answer_a": case[f"{a_model}_reply"], "answer_b": case[f"{b_model}_reply"],
               "winner": "", "reason": ""}
        rows.append(row)
        mappings.append({"review_id": str(index), "source_id": case["id"],
                         "prompt_key": case["prompt_key"], "A": a_model, "B": b_model,
                         "content_sha256": content_digest(row)})
    Path(form_path).parent.mkdir(parents=True, exist_ok=True)
    # BOM allows spreadsheet applications to detect Chinese UTF-8 text correctly.
    with Path(form_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(mapping_path, {"schema_version": 1, "seed": seed, "cases_path": str(cases_path),
                              "cases_sha256": sha256_file(cases_path), "cases": mappings})
    print(f"已准备 {len(rows)} 条盲评 -> {form_path}")
    print("仅填写 winner（A/B/tie）和 reason；判分完成前不要查看 mapping 文件。")


def summarize(form_path, mapping_path, output_path, overwrite=False,
              reviewer_type="unspecified", review_provenance=None):
    ensure_outputs_available([output_path], overwrite)
    provenance = {}
    if review_provenance:
        provenance = json.loads(Path(review_provenance).read_text(encoding="utf-8"))
        if provenance.get("scored_form_sha256") != sha256_file(form_path):
            raise ValueError("评审来源记录与当前评审表不一致")
        declared_type = provenance.get("reviewer_type", "unspecified")
        if reviewer_type != "unspecified" and reviewer_type != declared_type:
            raise ValueError("评审类型与来源记录不一致")
        reviewer_type = declared_type
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if mapping.get("schema_version") != 1:
        raise ValueError("不支持的盲评 mapping 版本")
    if sha256_file(mapping["cases_path"]) != mapping["cases_sha256"]:
        raise ValueError("原始生成对比已变化，与盲评 mapping 不一致")
    with Path(form_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS) <= set(reader.fieldnames or []):
            raise ValueError(f"评审表必须包含字段: {FIELDS}")
        rows = list(reader)
    expected = {item["review_id"]: item for item in mapping["cases"]}
    ids = [row["review_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("评审表样本缺失、重复或新增；必须评完原始全部样本")
    if any(not row["winner"].strip() for row in rows):
        count = sum(not row["winner"].strip() for row in rows)
        raise ValueError(f"尚有 {count} 条未判分，不生成正式胜率")
    counts = {"dpo_wins": 0, "sft_wins": 0, "ties": 0}
    judgments = []
    for row in rows:
        item = expected[row["review_id"]]
        if content_digest(row) != item["content_sha256"]:
            raise ValueError(f"review_id={row['review_id']} 的问题或答案被修改")
        label = row["winner"].strip().upper()
        if label not in {"A", "B", "TIE"}:
            raise ValueError(f"review_id={row['review_id']} 的 winner 必须为 A、B 或 tie")
        winner = "tie" if label == "TIE" else item[label]
        counts["ties" if winner == "tie" else f"{winner}_wins"] += 1
        judgments.append({"review_id": row["review_id"], "source_id": item["source_id"],
                          "prompt_key": item["prompt_key"], "winner": winner,
                          "reason": row["reason"].strip()})
    total = len(rows)
    decisive = counts["dpo_wins"] + counts["sft_wins"]
    rate, interval = None, None
    if decisive:
        rate = counts["dpo_wins"] / decisive
        z = 1.959963984540054
        denominator = 1 + z * z / decisive
        center = (rate + z * z / (2 * decisive)) / denominator
        margin = z * math.sqrt(rate * (1 - rate) / decisive + z * z / (4 * decisive**2)) / denominator
        interval = [center - margin, center + margin]
    result = {
        "schema_version": 1, "samples": total, **counts,
        "reviewer_type": reviewer_type,
        "independent_human_review": provenance.get("independent_human_review"),
        "review_context_limit": provenance.get("blinded_review_limit"),
        "human_medical_validation_status": provenance.get("human_medical_validation_status", "unspecified"),
        "review_provenance_path": str(review_provenance) if review_provenance else None,
        "review_provenance_sha256": sha256_file(review_provenance) if review_provenance else None,
        "dpo_win_rate_excluding_ties": rate,
        "dpo_win_rate_all_samples": counts["dpo_wins"] / total,
        "dpo_score_ties_half": (counts["dpo_wins"] + 0.5 * counts["ties"]) / total,
        "wilson95_excluding_ties": interval,
        "form_sha256": sha256_file(form_path), "mapping_sha256": sha256_file(mapping_path),
        "cases_sha256": mapping["cases_sha256"], "judgments": judgments,
    }
    write_json(output_path, result)
    print(f"DPO 胜 {counts['dpo_wins']} / 平 {counts['ties']} / SFT 胜 {counts['sft_wins']}")
    print(f"排除平局胜率: {rate:.2%}" if rate is not None else "全部平局，无排除平局胜率")
    print(f"平局计半分: {result['dpo_score_ties_half']:.2%}")
    print(f"逐条判分及统计 -> {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--cases", default="out/generation_test.json")
    prepare_parser.add_argument("--seed", type=int, default=42)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output", default="out/preference_review.summary.json")
    summarize_parser.add_argument("--reviewer-type", choices=["unspecified", "human", "model_review", "ai_assisted", "mixed"],
                                  default="unspecified")
    summarize_parser.add_argument("--review-provenance", help="评审来源 JSON，必须记录当前表格 SHA-256")
    for subparser in (prepare_parser, summarize_parser):
        subparser.add_argument("--form", default=DEFAULT_FORM)
        subparser.add_argument("--mapping", default=DEFAULT_MAPPING)
        subparser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.cases, args.form, args.mapping, args.seed, args.overwrite)
    else:
        summarize(args.form, args.mapping, args.output, args.overwrite,
                  args.reviewer_type, args.review_provenance)


if __name__ == "__main__":
    main()
