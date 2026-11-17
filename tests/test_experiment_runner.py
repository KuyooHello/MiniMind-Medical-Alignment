import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data_splits import sha256_file, write_json
from scripts.run_experiments import train


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.out = self.root / "out/experiments"
        self.out.mkdir(parents=True)
        (self.root / "dataset").mkdir()
        (self.root / "dataset/original.jsonl").write_text("original data")
        self.filtered = self.root / "dataset/filtered.jsonl"
        self.filtered.write_text("filtered data")
        checkpoint = self.out / "filtered_control_768.pth"
        checkpoint.write_bytes(b"completed checkpoint")
        self.settings = {"data_path": "../dataset/original.jsonl", "seed": 42}
        self.extra = {"data_path": "../dataset/filtered.jsonl", "nll_weight": .1}
        write_json(self.out / "filtered_control_768.training.json", {
            "completed": True, "settings": {**self.settings, **self.extra},
            "data_sha256": sha256_file(self.filtered),
            "checkpoint_sha256": sha256_file(checkpoint)})

    def run_cached(self):
        with patch("scripts.run_experiments.ROOT", self.root), \
                patch("scripts.run_experiments.OUT", self.out), \
                patch("scripts.run_experiments.subprocess.run") as process:
            train("filtered_control", "unused.py", self.settings, self.extra)
            process.assert_not_called()

    def test_completed_control_checks_its_overridden_dataset(self):
        self.run_cached()

    def test_completed_control_rejects_changed_filtered_data(self):
        self.filtered.write_text("changed filtered data")
        with self.assertRaisesRegex(RuntimeError, "数据发生变化"):
            self.run_cached()


if __name__ == "__main__":
    unittest.main()
