import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_preferences import audit
from scripts.data_splits import sha256_file, split_by_prompt, write_json, write_jsonl


class PreferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rows = [{"conversations": [{"role": "user", "content": f"测试问题 {index}"},
                                    {"role": "assistant", "content": "参考内容"}]}
                for index in range(20)]
        splits = split_by_prompt(rows)
        manifest = {"schema_version": 1, "splits": {}}
        for name, items in splits.items():
            path = self.root / f"{name}.jsonl"
            write_jsonl(path, items)
            manifest["splits"][name] = {"rows": len(items), "sha256": sha256_file(path)}
        write_json(self.root / "manifest.json", manifest)
        self.input_path = self.root / "pairs.jsonl"
        pairs = [{"chosen": row["conversations"],
                  "rejected": [row["conversations"][0], {"role": "assistant", "content": "采样内容"}]}
                 for row in splits["train"][:3]]
        write_jsonl(self.input_path, pairs)
        self.review = {"input_sha256": sha256_file(self.input_path), "selection_seed": 42,
                       "sample_size": 2, "judgments": [
                           {"audit_id": index, "preference": "reference_preferred",
                            "reference_major_issue": False, "retain_in_filtered_control": True}
                           for index in (1, 2)]}

    def run_review(self):
        path = self.root / "review.json"
        write_json(path, self.review)
        return audit(self.input_path, self.root, self.root / "filtered.jsonl",
                     self.root / "sample.json", self.root / "report.json", 2, 42, path)

    def test_valid_review_removes_only_the_sampled_excluded_pair(self):
        self.review["judgments"][0].update(preference="uncertain", retain_in_filtered_control=False)
        result = self.run_review()
        self.assertEqual(result["retained_pairs"], 2)
        self.assertEqual(len(result["semantic_excluded_pairs"]), 1)

    def test_changed_input_cannot_reuse_judgments(self):
        with self.input_path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "哈希不匹配"):
            self.run_review()

    def test_duplicate_judgment_is_rejected(self):
        self.review["judgments"].append(dict(self.review["judgments"][0]))
        with self.assertRaisesRegex(ValueError, "重复"):
            self.run_review()

    def test_reversed_preference_cannot_be_retained(self):
        self.review["judgments"][0]["preference"] = "sampled_preferred"
        with self.assertRaisesRegex(ValueError, "不可保留"):
            self.run_review()


if __name__ == "__main__":
    unittest.main()
