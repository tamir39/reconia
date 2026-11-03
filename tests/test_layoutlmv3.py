import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from models.layoutlmv3 import train as trainer
from models.layoutlmv3.dataset import (
    LayoutLMv3DataCollator,
    LayoutLMv3Dataset,
    decode_predictions,
)


class LayoutLMv3Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.weights_dir = self.root / "weights" / "layoutlmv3"
        self.weights_dir.mkdir(parents=True)
        tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + sorted(ByteLevel.alphabet())
        vocab = {token: index for index, token in enumerate(tokens)}
        (self.weights_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (self.weights_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        tokenizer = LayoutLMv3TokenizerFast(
            vocab_file=str(self.weights_dir / "vocab.json"),
            merges_file=str(self.weights_dir / "merges.txt"),
        )
        self.processor = LayoutLMv3Processor(
            LayoutLMv3ImageProcessor(apply_ocr=False, size={"height": 32, "width": 32}),
            tokenizer,
        )
        self.processor.save_pretrained(self.weights_dir)
        self.labels = {"O": 0, "B-item": 1, "I-item": 2}
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()
        (self.cache_dir / "labels.json").write_text(json.dumps(list(self.labels)), encoding="utf-8")
        Image.new("RGB", (80, 100), "white").save(self.root / "receipt.png")

    def write_sample(self, split, count):
        sample = {
            "id": split,
            "image_path": "receipt.png",
            "ocr": [
                {"text": "tea", "box": [10, 20, 200, 100],
                 "label": "B-item" if index == 0 else "I-item"}
                for index in range(count)
            ],
            "gt": {"item": [" ".join(["tea"] * count)]},
        }
        path = self.cache_dir / f"{split}.jsonl"
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        return path

    def test_images_boxes_and_completed_baseline_truncation(self):
        path = self.write_sample("val", 180)
        self.processor.image_processor.apply_ocr = True
        dataset = LayoutLMv3Dataset(path, self.processor, self.labels, self.root)
        self.assertFalse(self.processor.image_processor.apply_ocr)
        features = [dataset[index] for index in range(len(dataset))]
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]["input_ids"]), 512)
        kept_words = sum(label != -100 for label in features[0]["labels"])
        self.assertLess(kept_words, 180)
        batch = LayoutLMv3DataCollator(self.processor.tokenizer)(features)
        self.assertEqual(batch["bbox"].dtype, torch.long)
        self.assertEqual(tuple(batch["pixel_values"].shape), (len(features), 3, 32, 32))
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        for row in features:
            self.assertTrue(all(box == [10, 20, 200, 100] for box, label in zip(row["bbox"], row["labels"]) if label != -100))
        predictions = [[max(label, 0) for label in row["labels"]] for row in features]
        decoded = decode_predictions(
            predictions, dataset.samples, self.processor.tokenizer,
            {value: key for key, value in self.labels.items()},
        )
        self.assertEqual(decoded, [{"item": [" ".join(["tea"] * kept_words)]}])
        self.assertEqual(dataset.samples[0]["gt"], {"item": [" ".join(["tea"] * 180)]})

    def test_train_resume_and_separate_test_with_real_tiny_model(self):
        for split in ("train", "val", "test"):
            self.write_sample(split, 2)
        config = LayoutLMv3Config(
            vocab_size=len(self.processor.tokenizer), hidden_size=24,
            num_hidden_layers=1, num_attention_heads=4, intermediate_size=32,
            coordinate_size=4, shape_size=4, input_size=32, patch_size=16,
            num_labels=len(self.labels),
        )
        LayoutLMv3ForTokenClassification(config).save_pretrained(self.weights_dir)
        with (
            patch.object(trainer, "verify_baseline_data"),
            patch.object(trainer, "get_repo_dir", return_value=self.root),
            patch.object(trainer, "get_cache_dir", return_value=self.cache_dir),
            patch.object(trainer, "get_data_dir", return_value=self.root),
            patch.object(trainer, "load_labels", return_value=list(self.labels)),
        ):
            trainer.train_model("sroie", epochs=1, batch_size=1, min_epochs=1)
            checkpoint_dir = self.root / "checkpoints/sroie/layoutlmv3"
            out_dir = self.root / "results/sroie/layoutlmv3"
            checkpoint = torch.load(checkpoint_dir / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["best_epoch"], 1)
            self.assertEqual(len(checkpoint["history"]), 1)
            self.assertTrue((checkpoint_dir / "best_weights.pt").exists())
            self.assertTrue((out_dir / "history.csv").exists())
            self.assertFalse((out_dir / "test_metrics.json").exists())
            with self.assertRaises(FileExistsError):
                trainer.train_model("sroie", epochs=1)
            trainer.train_model("sroie", epochs=1, resume=True)
            trainer.train_model("sroie", epochs=2, batch_size=1, min_epochs=1, resume=True)
            resumed = torch.load(checkpoint_dir / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual([row["epoch"] for row in resumed["history"]], [1, 2])
            best_before = (checkpoint_dir / "best_weights.pt").read_bytes()
            trainer.train_model("sroie", batch_size=1, test_only=True)
            result = json.loads((out_dir / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["model"], "layoutlmv3")
            self.assertEqual(result["dataset"], "sroie")
            self.assertTrue(all(0 <= result[key] <= 1 for key in ("precision", "recall", "f1")))
            self.assertEqual(best_before, (checkpoint_dir / "best_weights.pt").read_bytes())


if __name__ == "__main__":
    unittest.main()
