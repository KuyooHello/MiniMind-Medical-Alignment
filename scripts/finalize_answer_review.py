"""Lock anonymous judgments before revealing identities, then publish the paired review."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.blind_review import FIELDS, content_digest, summarize
from scripts.data_splits import ensure_outputs_available, sha256_file, write_json


TIE_CATEGORIES = {"similar_quality", "both_poor", "uncertain"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_form(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("评审表字段不匹配")
        rows = list(reader)
    ids = [row["review_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("评审表为空或包含重复 id")
    return rows


def validate_judgments(rows, judgments):
    ids = [item.get("review_id") for item in judgments]
    if len(set(ids)) != len(ids) or set(ids) != {row["review_id"] for row in rows}:
        raise ValueError("必须完整判分，id 不可缺失、重复或新增")
    for item in judgments:
        if item.get("winner") not in {"A", "B", "tie"} or not str(item.get("reason", "")).strip():
            raise ValueError("每条判断必须有合法 winner 和非空理由")
        category = item.get("tie_category")
        if (item["winner"] == "tie" and category not in TIE_CATEGORIES
                or item["winner"] != "tie" and category is not None):
            raise ValueError("平局分类与 winner 不一致")
        for key in ("a_major_issue", "b_major_issue"):
            if type(item.get(key)) is not bool:
                raise ValueError("明显问题标记必须为布尔值")
        if category == "both_poor" and not (item["a_major_issue"] and item["b_major_issue"]):
            raise ValueError("both_poor 必须同时标记两份答案有明显问题")
        if not isinstance(item.get("sources"), list) or any(
                not isinstance(url, str) or not url.startswith("https://") for url in item["sources"]):
            raise ValueError("sources 必须为 HTTPS 来源列表")
    indexed = {item["review_id"]: item for item in judgments}
    return [indexed[row["review_id"]] for row in rows]


def lock_review(directory):
    """This phase reads the anonymous form and protocol, NEVER cases or identity mapping."""
    directory = Path(directory)
    form_path, protocol_path = directory / "review.csv", directory / "protocol.json"
    outputs = [directory / name for name in (
        "review.scored.csv", "judgments.locked.json", "review.provenance.json", "judgments.lock.json")]
    ensure_outputs_available(outputs)
    protocol = read_json(protocol_path)
    if sha256_file(form_path) != protocol["unscored_form_sha256"]:
        raise ValueError("原始匿名表已改变")
    rows = read_form(form_path)
    if len(rows) != protocol["samples"] or any(row["winner"] or row["reason"] for row in rows):
        raise ValueError("原始表必须保持未判分状态且样本数与方案一致")
    batch_paths = sorted((directory / "judgments").glob("batch_*.json"))
    judgments = validate_judgments(rows, [item for path in batch_paths for item in read_json(path)])
    for row, item in zip(rows, judgments):
        row.update(winner=item["winner"], reason=item["reason"])
    with outputs[0].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(outputs[1], judgments)
    locked_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "reviewer_type": "model_review", "independent_human_review": False,
        "human_medical_validation_status": "not_performed",
        "scored_form_sha256": sha256_file(outputs[0]), "judgments_locked_at_utc": locked_at,
        "blinded_review_limit": (
            "One model reviewer with project context. Previously exposed questions were excluded. "
            "Only anonymous A/B answers were inspected for these new questions; the identity mapping "
            "and model-labelled new outputs were not inspected until all judgments were locked. "
            "This is not an independent human or clinical expert assessment."),
        "reference_answers_shown": False,
        "evidence_policy": "Primary medical sources were checked for substantive factual judgments; URLs are recorded per item.",
        "major_issue_definition": "Reviewer-assessed substantial factual/safety error, fabricated case facts, or serious failure to address the task. Not a clinical accuracy label.",
        "tie_category_definition": {
            "similar_quality": "No clear relative advantage; this does not certify either answer as complete or correct.",
            "both_poor": "Both answers contain comparable major problems without a clear relative advantage.",
            "uncertain": "Evidence is insufficient to rank the answers reliably."},
    }
    write_json(outputs[2], provenance)
    tracked = [form_path, protocol_path, *batch_paths, *outputs[:3]]
    lock = {
        "schema_version": 1, "samples": len(rows), "locked_at_utc": locked_at,
        "files": {str(path.relative_to(directory)): sha256_file(path) for path in tracked},
        "prepared_cases_sha256": protocol["cases_sha256"],
        "prepared_mapping_sha256": protocol["private_mapping_sha256"],
        "identities_read_in_lock_phase": False,
    }
    # Last write: publication refuses absent or inconsistent locks. Local content hashes
    # make subsequent edits detectable; they are not an external timestamp or signature.
    write_json(outputs[3], lock)
    print(f"已锁定 {len(rows)} 条匿名判断；本步骤未读取身份映射。")
    return lock


def verify_lock(directory):
    directory = Path(directory)
    lock = read_json(directory / "judgments.lock.json")
    for relative, expected_hash in lock["files"].items():
        if sha256_file(directory / relative) != expected_hash:
            raise ValueError(f"锁定后内容已改变: {relative}")
    rows = read_form(directory / "review.scored.csv")
    judgments = validate_judgments(rows, read_json(directory / "judgments.locked.json"))
    if len(rows) != lock["samples"] or any(
            row["winner"] != item["winner"] or row["reason"] != item["reason"]
            for row, item in zip(rows, judgments)):
        raise ValueError("锁定表格与逐条判断不一致")
    return lock, rows, judgments


def publish_review(directory, results_dir="results", examples_dir="examples", overwrite=False):
    directory, results_dir, examples_dir = map(Path, (directory, results_dir, examples_dir))
    lock, rows, judgments = verify_lock(directory)
    # No access to either file occurs before verify_lock succeeds.
    cases_path, mapping_path = directory / "cases.json", directory / "mapping.private.json"
    if sha256_file(cases_path) != lock["prepared_cases_sha256"] or sha256_file(mapping_path) != lock["prepared_mapping_sha256"]:
        raise ValueError("生成答案或身份映射与生成时的冻结记录不一致")
    protocol = read_json(directory / "protocol.json")
    mapping = read_json(mapping_path)
    cases = read_json(cases_path)
    cases_by_id = {case["id"]: case for case in cases}
    mapping_by_id = {item["review_id"]: item for item in mapping["cases"]}
    if len(mapping_by_id) != len(rows) or set(mapping_by_id) != {row["review_id"] for row in rows}:
        raise ValueError("身份映射 id 不完整")
    if set(protocol["prompt_keys"]) != {item["prompt_key"] for item in mapping["cases"]}:
        raise ValueError("抽样问题与方案不一致")
    for row in rows:
        if content_digest(row) != mapping_by_id[row["review_id"]]["content_sha256"]:
            raise ValueError("匿名问题或答案内容已改变")
    outputs = {
        "form": examples_dir / "answer_review.csv",
        "summary": results_dir / "answer_review.json",
        "protocol": results_dir / "answer_review_protocol.json",
        "provenance": results_dir / "answer_review_provenance.json",
        "judgments": results_dir / "answer_review_judgments.json",
        "lock": results_dir / "answer_review_lock.json",
    }
    ensure_outputs_available(list(outputs.values()), overwrite)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["form"].write_bytes((directory / "review.scored.csv").read_bytes())
    outputs["provenance"].write_bytes((directory / "review.provenance.json").read_bytes())
    summary = summarize(outputs["form"], mapping_path, outputs["summary"], overwrite,
                        reviewer_type="model_review", review_provenance=outputs["provenance"])
    rich, issue_counts, positions = [], Counter(), Counter()
    for item in judgments:
        identity = mapping_by_id[item["review_id"]]
        positions[f"dpo_on_{'A' if identity['A'] == 'dpo' else 'B'}"] += 1
        flags = {identity["A"]: item["a_major_issue"], identity["B"]: item["b_major_issue"]}
        issue_counts.update(model for model, flagged in flags.items() if flagged)
        rich.append({**item, "source_id": identity["source_id"], "prompt_key": identity["prompt_key"],
                     "model_a": identity["A"], "model_b": identity["B"],
                     "model_winner": "tie" if item["winner"] == "tie" else identity[item["winner"]],
                     "major_issue_by_model": flags,
                     "generation_by_model": cases_by_id[identity["source_id"]]["generation"]})
    stopping = {}
    for model in ("sft", "dpo"):
        records = [case["generation"][model] for case in cases]
        stopping[model] = {
            "samples": len(records), "eos_seen_count": sum(record["eos_seen"] for record in records),
            "cap_without_eos_count": sum(record["generation_cap_without_eos"] for record in records),
            "mean_generated_tokens_including_eos": sum(record["generated_tokens"] for record in records) / len(records),
        }
    summary.update({
        "tie_categories": dict(Counter(item["tie_category"] for item in judgments if item["winner"] == "tie")),
        "major_issue_flagged_answers": {model: issue_counts[model] for model in ("sft", "dpo")},
        "both_major_issue_pairs": sum(item["a_major_issue"] and item["b_major_issue"] for item in judgments),
        "major_issue_interpretation": "Model reviewer flags, not clinical accuracy or expert-validated safety rates.",
        "position_allocation": dict(positions), "generation_stopping": stopping,
        "all_selected_pairs_judged": len(rows) == protocol["samples"], "pairs_removed_after_generation": 0,
        "previously_exposed_questions_in_selected_set": protocol["previously_exposed_questions_in_selected_set"],
        "protocol_path": str(outputs["protocol"]), "judgments_path": str(outputs["judgments"]),
        "judgment_lock_sha256": sha256_file(directory / "judgments.lock.json"),
        "interpretation": "Relative preference under this frozen sample, decoding configuration, and one model reviewer. The interval reflects decisive-case sampling uncertainty, not reviewer bias or clinical correctness.",
    })
    write_json(outputs["summary"], summary)
    write_json(outputs["judgments"], rich)
    write_json(outputs["protocol"], {**protocol, "review_status": "completed",
                                    "allocation_seed_disclosed_after_lock": mapping["seed"],
                                    "judgment_lock_sha256": sha256_file(directory / "judgments.lock.json"),
                                    "scored_form_sha256": sha256_file(outputs["form"])})
    outputs["lock"].write_bytes((directory / "judgments.lock.json").read_bytes())
    print(f"全部逐条判断、匿名答案、方案和统计已导出至 {results_dir} / {examples_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["lock", "publish"])
    parser.add_argument("--directory", default="out/answer_blind_review")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--examples-dir", default="examples")
    parser.add_argument("--overwrite", action="store_true", help="仅允许重新导出已锁定的结果，不允许重新判分")
    args = parser.parse_args()
    if args.phase == "lock":
        lock_review(args.directory)
    else:
        publish_review(args.directory, args.results_dir, args.examples_dir, args.overwrite)


if __name__ == "__main__":
    main()
