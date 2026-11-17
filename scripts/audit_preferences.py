"""Audit all preference pairs with explicit structural and response-quality rules."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from scripts.data_splits import (
    load_verified_splits, normalize_prompt, prompt_key, read_jsonl, sha256_file,
    validate_splits, write_json, write_jsonl,
)


def repeat_fraction(text, width=4):
    text = re.sub(r"\s+", "", text)
    grams = [text[index:index + width] for index in range(max(0, len(text) - width + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def quality_flags(pair):
    question = next(message["content"] for message in pair["chosen"] if message["role"] == "user")
    chosen = pair["chosen"][-1]["content"].strip()
    rejected = pair["rejected"][-1]["content"].strip()
    flags = []
    if not chosen or not rejected:
        flags.append("empty_response")
    if normalize_prompt(chosen) == normalize_prompt(rejected):
        flags.append("identical_responses")
    if "\ufffd" in question + chosen:
        flags.append("corrupt_prompt_or_reference")
    if re.search(r"图片涉及用户隐私|如图|见图|附图|看图|看片|发.{0,3}[xX]光", question):
        flags.append("unavailable_image_context")
    if re.search(r"您的问题是关于|请问您是想|需要了解具体症状", question):
        flags.append("assistant_style_prompt")
    if re.search(r"(.{2,40}?)\1{3,}", re.sub(r"\s+", "", chosen)):
        flags.append("reference_consecutive_repetition")
    if len(chosen) > 100 and repeat_fraction(chosen) > 0.45:
        flags.append("reference_high_repetition")
    # These are narrow, auditable sentinel checks, not a clinical safety classifier.
    if re.search(r"口服甲醛|甲醛胶囊|抗生素[，、：:\s]*(?:药物)?[，、：:\s]*(?:如)?(?:布洛芬|对乙酰氨基酚)", chosen):
        flags.append("reference_known_unsafe_or_wrong_drug_class")
    return flags


def audit(input_path, split_dir, output_path, review_path, report_path, sample_size=50, seed=20261117,
          semantic_review=None):
    if sample_size < 1:
        raise ValueError("抽样数量必须大于 0")
    rows = read_jsonl(input_path)
    splits, _ = load_verified_splits(split_dir)
    validate_splits(splits, rows)
    flags = [quality_flags(row) for row in rows]
    counts = Counter(flag for group in flags for flag in group)
    retained = [row for row, group in zip(rows, flags) if not group]
    rule_retained_count = len(retained)
    sample_indices = random.Random(seed).sample(range(len(rows)), min(sample_size, len(rows)))
    review = [
        {"audit_id": index + 1, "pair_index": source_index, "prompt_key": prompt_key(rows[source_index]["chosen"]),
         "prompt": rows[source_index]["chosen"][0]["content"],
         "reference": rows[source_index]["chosen"][-1]["content"],
         "sampled": rows[source_index]["rejected"][-1]["content"],
         "rule_flags": flags[source_index]}
        for index, source_index in enumerate(sample_indices)
    ]
    semantic_excluded = set()
    semantic = None
    if semantic_review:
        semantic = json.loads(Path(semantic_review).read_text(encoding="utf-8"))
        if semantic.get("input_sha256") != sha256_file(input_path):
            raise ValueError("语义审阅对应的数据哈希不匹配，禁止套用已有标签")
        if semantic["selection_seed"] != seed or semantic["sample_size"] != len(review):
            raise ValueError("语义审阅与当前抽样设置不一致")
        by_id = {row["audit_id"]: row for row in review}
        judgments = semantic["judgments"]
        if len(judgments) != len(by_id) or {row["audit_id"] for row in judgments} != set(by_id):
            raise ValueError("语义审阅缺失、重复或新增样本")
        for row in judgments:
            if (row["preference"] not in {"reference_preferred", "sampled_preferred", "uncertain"}
                    or type(row["retain_in_filtered_control"]) is not bool
                    or type(row["reference_major_issue"]) is not bool):
                raise ValueError("语义审阅的偏好标签或布尔字段不合法")
            if row["retain_in_filtered_control"] and (
                    row["preference"] != "reference_preferred" or row["reference_major_issue"]):
                raise ValueError("不可保留标签反转、不确定或参考答案存在重大问题的样本")
        semantic_excluded = {by_id[row["audit_id"]]["pair_index"] for row in semantic["judgments"]
                             if not row["retain_in_filtered_control"]}
        retained = [row for index, (row, group) in enumerate(zip(rows, flags))
                    if not group and index not in semantic_excluded]
    write_jsonl(output_path, retained)
    write_json(review_path, review)
    report = {
        "schema_version": 1, "input_pairs": len(rows), "retained_pairs_before_semantic_review": rule_retained_count,
        "removed_pairs_by_rules": len(rows) - rule_retained_count, "rule_flag_counts": dict(counts),
        "sample_size": len(review), "sample_seed": seed,
        "sampling": "uniform random sample of all original pairs, selected before semantic review",
        "input_sha256": sha256_file(input_path), "output_sha256": sha256_file(output_path),
        "filtered_dataset_path": str(output_path), "review_sample_path": str(review_path),
        "pair_decisions": [{"pair_index": index, "prompt_key": prompt_key(row["chosen"]), "rule_flags": group}
                           for index, (row, group) in enumerate(zip(rows, flags))],
        "clinical_validation": "Not performed by an independent medical expert. Structural and textual heuristics do not certify clinical correctness.",
    }
    if semantic is not None:
        report.update({"retained_pairs": len(retained), "removed_pairs_total": len(rows) - len(retained),
                       "semantic_review_sha256": sha256_file(semantic_review),
                       "semantic_review_outcomes": dict(Counter(row["preference"] for row in semantic["judgments"])),
                       "audited_reference_major_issue_count": sum(row["reference_major_issue"] for row in semantic["judgments"]),
                       "semantic_excluded_pairs": sorted(semantic_excluded),
                       "filtered_dataset_sha256": sha256_file(output_path)})
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/dpo_medical.jsonl")
    parser.add_argument("--split-dir", default="dataset/medical")
    parser.add_argument("--output", default="dataset/dpo_medical_filtered.jsonl")
    parser.add_argument("--review-output", default="out/preference_audit_sample.json")
    parser.add_argument("--report-output", default="out/preference_quality.json")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--semantic-review", help="可选：重放与输入哈希绑定的已完成抽样审阅")
    args = parser.parse_args()
    report = audit(args.input, args.split_dir, args.output, args.review_output, args.report_output,
                   args.sample_size, args.seed, args.semantic_review)
    print(json.dumps({key: value for key, value in report.items() if key != "pair_decisions"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
