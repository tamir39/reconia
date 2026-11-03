import hashlib
import importlib
import json
import random
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.baseline_protocol import canonical_records, load_labels, verify_baseline_data
from utils.metrics import evaluate_dataset


class BaselineProtocolTests(unittest.TestCase):
    def test_new_trainers_reject_changed_data_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ["train", "val", "test"]:
                (root / f"{split}.jsonl").write_text(json.dumps({
                    "id": "changed", "image_path": "train/img/changed.jpg", "ocr": [], "gt": {}})
                    + "\n", encoding="utf-8")
            for name in ["layoutlmv3", "donut", "qwen_vl"]:
                trainer = importlib.import_module(f"models.{name}.train")
                loader = (patch.object(trainer.AutoProcessor, "from_pretrained") if name == "layoutlmv3"
                          else patch.object(trainer, "build_model"))
                with (self.subTest(model=name),
                      patch.object(trainer, "get_repo_dir", return_value=root),
                      patch.object(trainer, "get_cache_dir", return_value=root), loader as model_loader):
                    with self.assertRaisesRegex(ValueError, "differs from completed baselines"):
                        trainer.train_model("sroie", epochs=1)
                    model_loader.assert_not_called()

    def test_scores_equal_the_frozen_completed_run_scorer(self):
        reference = runpy.run_path(str(Path(__file__).parent / "fixtures/completed_baseline_metrics.py"))
        old_score = reference["evaluate_dataset"]
        examples = [
            ([{"item": ["Tea", "Tea"], "total": [" 100 "]}],
             [{"item": [" tea "], "total": ["100"], "extra": ["wrong"]}]),
            ([{"total": ["21.60"]}], [{"total": ["21.6"]}]),
            ([{"total": ["100"]}, {"total": ["200"]}], [{}, {"total": ["200", "200"]}]),
        ]
        rng = random.Random(42)
        for _ in range(100):
            gts, preds = [], []
            for _ in range(rng.randint(1, 8)):
                gts.append({field: rng.choices([" Tea ", "100", "100", "", "A  B"], k=rng.randint(0, 4))
                            for field in ["item", "total"]})
                preds.append({field: rng.choices(["tea", "100", "A B", "WRONG", ""], k=rng.randint(0, 4))
                              for field in rng.sample(["item", "total", "extra"], rng.randint(0, 3))})
            examples.append((gts, preds))
        for gts, preds in examples:
            self.assertEqual(evaluate_dataset(gts, preds), old_score(gts, preds))

    def test_frozen_label_inventory_does_not_invent_i_labels(self):
        self.assertEqual(len(load_labels("sroie")), 9)
        self.assertEqual(len(load_labels("cord_v2")), 49)
        self.assertEqual(len(load_labels("wildreceipt")), 13)
        self.assertFalse(any(label.startswith("I-") for label in load_labels("wildreceipt")))

    def test_contract_accepts_line_endings_and_cord_root_but_rejects_changed_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [{"id": "0", "image_path": "images/train/0.png", "ocr": [],
                        "gt": {"total": ["100"]}}]
            digest = hashlib.sha256(canonical_records(samples).encode("utf-8")).hexdigest()
            contract = {"datasets": {"cord_v2": {"splits": {
                split: {"count": 1, "sha256": digest} for split in ["train", "val", "test"]}}}}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(contract), encoding="utf-8")
            samples[0]["image_path"] = "D:/portable/cache/images/train/0.png"
            for split in ["train", "val", "test"]:
                (root / f"{split}.jsonl").write_bytes((json.dumps(samples[0]) + "\r\n").encode("utf-8"))
            self.assertEqual(verify_baseline_data("cord_v2", root, manifest)["train"]["count"], 1)
            samples[0]["gt"]["total"] = ["999"]
            (root / "test.jsonl").write_text(json.dumps(samples[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from completed baselines"):
                verify_baseline_data("cord_v2", root, manifest)


if __name__ == "__main__":
    unittest.main()
