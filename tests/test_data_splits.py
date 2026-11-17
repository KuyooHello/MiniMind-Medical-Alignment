import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_splits import (
    ensure_outputs_available, load_verified_splits, normalize_prompt, prompt_key,
    sha256_file, split_by_prompt, validate_splits, write_json, write_jsonl,
)


def sample(question, answer="参考回答"):
    return {"conversations": [{"role": "user", "content": question},
                              {"role": "assistant", "content": answer}]}


class DataSplitTests(unittest.TestCase):
    def setUp(self):
        self.samples = [sample(f"病例问题 {index}") for index in range(20)]
        self.splits = split_by_prompt(self.samples)

    def test_normalization_preserves_medical_numbers_and_negation(self):
        self.assertEqual(normalize_prompt(" 患者：ＡＢＣ\n 检查 "), "ABC 检查")
        self.assertNotEqual(prompt_key(sample("服用 1 mg")["conversations"]),
                            prompt_key(sample("服用 10 mg")["conversations"]))
        self.assertNotEqual(prompt_key(sample("没有发热")["conversations"]),
                            prompt_key(sample("有发热")["conversations"]))

    def test_same_question_with_different_answers_stays_in_one_split(self):
        rows = self.samples + [sample("病例问题 0", "另一种回答")]
        splits = split_by_prompt(rows)
        key = prompt_key(rows[0]["conversations"])
        owners = [name for name, items in splits.items()
                  if any(prompt_key(row["conversations"]) == key for row in items)]
        self.assertEqual(len(owners), 1)
        self.assertEqual(sum(len(rows) for rows in splits.values()), len(rows))

    def test_split_is_reproducible_when_source_order_changes(self):
        self.assertEqual(split_by_prompt(self.samples), split_by_prompt(list(reversed(self.samples))))

    def test_validation_rejects_same_question_even_when_answers_differ(self):
        contaminated = {name: list(rows) for name, rows in self.splits.items()}
        question = contaminated["train"][0]["conversations"][0]["content"]
        contaminated["test"].append(sample(question, "未见过的另一个答案"))
        with self.assertRaisesRegex(ValueError, "数据泄漏"):
            validate_splits(contaminated)

    def test_dpo_must_use_training_prompts_and_match_both_sides(self):
        train = self.splits["train"][0]["conversations"]
        test = self.splits["test"][0]["conversations"]
        validate_splits(self.splits, [{"chosen": train, "rejected": train}])
        with self.assertRaisesRegex(ValueError, "不属于训练集"):
            validate_splits(self.splits, [{"chosen": test, "rejected": test}])
        with self.assertRaisesRegex(ValueError, "问题不一致"):
            validate_splits(self.splits, [{"chosen": train, "rejected": test}])

    def test_saved_split_rejects_modified_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"schema_version": 1, "splits": {}}
            for name, rows in self.splits.items():
                path = root / f"{name}.jsonl"
                write_jsonl(path, rows)
                manifest["splits"][name] = {"rows": len(rows), "sha256": sha256_file(path)}
            write_json(root / "manifest.json", manifest)
            loaded, _ = load_verified_splits(root)
            self.assertEqual(loaded, self.splits)
            with (root / "test.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample("新增问题"), ensure_ascii=False) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest 不一致"):
                load_verified_splits(root)

    def test_invalid_ratios_and_empty_splits_are_rejected(self):
        with self.assertRaises(ValueError):
            split_by_prompt(self.samples, val_ratio=0.6, test_ratio=0.5)
        with self.assertRaises(ValueError):
            split_by_prompt(self.samples[:2])
        with self.assertRaises(ValueError):
            validate_splits({"train": self.samples, "val": [], "test": []})

    def test_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.jsonl"
            path.write_text("old data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_outputs_available([path])
            ensure_outputs_available([path], overwrite=True)


if __name__ == "__main__":
    unittest.main()
