import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.blind_review import prepare, summarize
from scripts.data_splits import sha256_file, write_json


class BlindReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "cases.json"
        self.form = self.root / "review.csv"
        self.mapping = self.root / "mapping.json"
        self.output = self.root / "summary.json"
        write_json(self.source, [
            {"id": index, "prompt_key": f"question-{index}", "prompt": f"问题 {index}",
             "sft_reply": f"SFT 回复 {index}", "dpo_reply": f"DPO 回复 {index}"}
            for index in range(1, 51)
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.source, self.form, self.mapping)

    def rows(self):
        with self.form.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def write_rows(self, rows):
        with self.form.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def fill(self, outcome):
        mapping = {row["review_id"]: row for row in json.loads(self.mapping.read_text())["cases"]}
        rows = self.rows()
        for row in rows:
            item = mapping[row["review_id"]]
            target = outcome(item["source_id"])
            row["winner"] = "tie" if target == "tie" else ("A" if item["A"] == target else "B")
        self.write_rows(rows)

    def summarize(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return summarize(self.form, self.mapping, self.output)

    def test_mapping_is_balanced_and_counts_decode_correctly(self):
        mapping = json.loads(self.mapping.read_text())["cases"]
        self.assertEqual(sum(row["A"] == "dpo" for row in mapping), 25)
        self.fill(lambda index: "dpo" if index <= 24 else ("tie" if index <= 33 else "sft"))
        result = self.summarize()
        self.assertEqual((result["dpo_wins"], result["ties"], result["sft_wins"]), (24, 9, 17))
        self.assertAlmostEqual(result["dpo_win_rate_excluding_ties"], 24 / 41)
        self.assertAlmostEqual(result["dpo_win_rate_all_samples"], 24 / 50)
        self.assertAlmostEqual(result["dpo_score_ties_half"], 0.57)

    def test_incomplete_reviews_do_not_create_a_result(self):
        with self.assertRaisesRegex(ValueError, "50 条未判分"):
            self.summarize()
        self.assertFalse(self.output.exists())

    def test_all_ties_have_no_decisive_win_rate(self):
        self.fill(lambda _: "tie")
        result = self.summarize()
        self.assertIsNone(result["dpo_win_rate_excluding_ties"])
        self.assertIsNone(result["wilson95_excluding_ties"])
        self.assertEqual(result["dpo_score_ties_half"], 0.5)

    def test_changed_answer_is_rejected(self):
        self.fill(lambda _: "dpo")
        rows = self.rows()
        rows[0]["answer_a"] = "被改写的答案"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "问题或答案被修改"):
            self.summarize()

    def test_dropped_review_is_rejected(self):
        self.fill(lambda _: "dpo")
        self.write_rows(self.rows()[:-1])
        with self.assertRaisesRegex(ValueError, "样本缺失"):
            self.summarize()

    def test_changed_source_is_rejected(self):
        self.fill(lambda _: "dpo")
        self.source.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "原始生成对比已变化"):
            self.summarize()

    def test_ai_review_type_is_retained_in_summary(self):
        self.fill(lambda _: "tie")
        provenance = self.root / "provenance.json"
        write_json(provenance, {"reviewer_type": "ai_assisted", "independent_human_review": False,
                                "scored_form_sha256": sha256_file(self.form)})
        with contextlib.redirect_stdout(io.StringIO()):
            result = summarize(self.form, self.mapping, self.output, review_provenance=provenance)
        self.assertEqual(result["reviewer_type"], "ai_assisted")
        self.assertFalse(result["independent_human_review"])
        self.assertEqual(result["review_provenance_sha256"], sha256_file(provenance))

    def test_review_cannot_be_mislabeled_as_human(self):
        self.fill(lambda _: "tie")
        provenance = self.root / "provenance.json"
        write_json(provenance, {"reviewer_type": "ai_assisted", "scored_form_sha256": sha256_file(self.form)})
        with self.assertRaisesRegex(ValueError, "评审类型与来源记录不一致"):
            summarize(self.form, self.mapping, self.output, reviewer_type="human", review_provenance=provenance)


if __name__ == "__main__":
    unittest.main()
