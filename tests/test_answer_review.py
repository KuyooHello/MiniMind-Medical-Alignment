"""Review completeness, pre-reveal locking, and faithful publication checks."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.blind_review import prepare
from scripts.data_splits import sha256_file, write_json
from scripts.finalize_answer_review import lock_review, publish_review, read_form, verify_lock


class AnswerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cases = self.directory / "cases.json"
        self.mapping = self.directory / "mapping.private.json"
        self.form = self.directory / "review.csv"
        self.batch = self.directory / "judgments" / "batch_01.json"
        write_json(self.cases, [{
            "id": index, "prompt_key": f"key-{index}", "prompt": f"问题 {index}\n补充",
            "sft_reply": f"第 {index} 个回答，含\"引号\"。", "dpo_reply": f"另一个回答\n{index}",
            "generation": {
                "sft": {"generated_tokens": 8, "eos_seen": True, "generation_cap_without_eos": False},
                "dpo": {"generated_tokens": 10, "eos_seen": False, "generation_cap_without_eos": True},
            },
        } for index in (1, 2)])
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cases, self.form, self.mapping, seed=83)
        write_json(self.directory / "protocol.json", {
            "samples": 2, "unscored_form_sha256": sha256_file(self.form),
            "cases_sha256": sha256_file(self.cases), "private_mapping_sha256": sha256_file(self.mapping),
            "prompt_keys": ["key-1", "key-2"], "previously_exposed_questions_in_selected_set": 0,
        })
        self.judgments = [
            {"review_id": "1", "winner": "A", "tie_category": None,
             "a_major_issue": False, "b_major_issue": True, "reason": "相对更相关", "sources": []},
            {"review_id": "2", "winner": "tie", "tie_category": "both_poor",
             "a_major_issue": True, "b_major_issue": True, "reason": "两者均有问题", "sources": []},
        ]
        write_json(self.batch, self.judgments)

    def lock(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return lock_review(self.directory)

    def publish(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return publish_review(self.directory, self.directory / "public_results", self.directory / "public_examples")

    def test_lock_does_not_need_identity_files_and_cannot_be_replaced(self):
        self.cases.unlink()
        self.mapping.unlink()
        lock = self.lock()
        self.assertFalse(lock["identities_read_in_lock_phase"])
        self.assertEqual(len(verify_lock(self.directory)[1]), 2)
        with self.assertRaises(FileExistsError):
            self.lock()

    def test_missing_duplicate_empty_reason_and_bad_ties_are_rejected(self):
        variants = [self.judgments[:1], self.judgments + self.judgments[:1]]
        for field, value in (("reason", ""), ("tie_category", "both_poor"), ("a_major_issue", "false")):
            rows = json.loads(json.dumps(self.judgments))
            rows[0][field] = value
            variants.append(rows)
        for records in variants:
            with self.subTest(records=records):
                write_json(self.batch, records)
                with self.assertRaises(ValueError):
                    self.lock()
                self.assertFalse((self.directory / "judgments.lock.json").exists())

    def test_changed_answer_is_rejected_before_access_to_mapping(self):
        self.lock()
        scored = self.directory / "review.scored.csv"
        scored.write_text(scored.read_text(encoding="utf-8-sig") + "altered", encoding="utf-8-sig")
        self.mapping.unlink()
        with self.assertRaisesRegex(ValueError, "锁定后内容已改变"):
            self.publish()
        self.assertFalse((self.directory / "public_results").exists())

    def test_changed_source_judgment_is_rejected_after_lock(self):
        self.lock()
        self.judgments[0]["winner"] = "B"
        write_json(self.batch, self.judgments)
        with self.assertRaisesRegex(ValueError, "锁定后内容已改变"):
            self.publish()

    def test_publication_preserves_answers_provenance_and_all_pairs(self):
        original = read_form(self.form)
        self.lock()
        result = self.publish()
        exported = read_form(self.directory / "public_examples" / "answer_review.csv")
        for before, after in zip(original, exported):
            for key in ("review_id", "prompt", "answer_a", "answer_b"):
                self.assertEqual(before[key], after[key])
        mapping = json.loads(self.mapping.read_text())["cases"]
        expected_winner = mapping[0]["A"]
        self.assertEqual(result[f"{expected_winner}_wins"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["tie_categories"], {"both_poor": 1})
        self.assertEqual(result["position_allocation"], {"dpo_on_A": 1, "dpo_on_B": 1})
        self.assertEqual(result["generation_stopping"]["dpo"]["cap_without_eos_count"], 2)
        self.assertEqual(result["generation_stopping"]["sft"]["eos_seen_count"], 2)
        self.assertEqual(result["reviewer_type"], "model_review")
        self.assertFalse(result["independent_human_review"])
        self.assertEqual(result["human_medical_validation_status"], "not_performed")
        self.assertTrue(result["all_selected_pairs_judged"])
        self.assertEqual(result["pairs_removed_after_generation"], 0)
        rich = json.loads((self.directory / "public_results" / "answer_review_judgments.json").read_text())
        self.assertEqual(len(rich), 2)
        self.assertEqual(rich[0]["model_winner"], expected_winner)
        self.assertEqual(rich[0]["major_issue_by_model"][expected_winner], False)
        self.assertIn("generation_by_model", rich[0])


if __name__ == "__main__":
    unittest.main()
