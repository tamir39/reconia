import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from utils.generation import parse_output
from utils.metrics import evaluate_dataset
from utils.token_data import TokenDataset, TokenCollator, decode_predictions
from utils.training import fit, load_weights


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, **kwargs):
        loss = self.weight.square()
        return (loss, None) if "images" in kwargs else SimpleNamespace(loss=loss)


class PipelineTests(unittest.TestCase):
    def test_macro_exact_values_and_duplicate_matching(self):
        result = evaluate_dataset(
            [{"total": ["100"], "item": ["Tea", "Tea"]}],
            [{"total": ["100"], "item": [" tea  "]}],
        )
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.75)
        self.assertAlmostEqual(result["f1"], 5 / 6)
        with self.assertRaises(ValueError):
            evaluate_dataset([{"total": ["100"]}], [])

    def test_generated_json_is_not_repaired_with_gt(self):
        self.assertEqual(parse_output('{"total":["100"]}'), {"total": ["100"]})
        self.assertEqual(parse_output('{"total":100}'), {})
        self.assertEqual(parse_output('{"total":['), {})

    def test_long_document_reassembles_without_duplicate_supervision(self):
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "tea": 4}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
        labels = {"O": 0, "B-item": 1, "I-item": 2}
        sample = {"id": "long", "ocr": [{"text": "tea", "box": [0, 0, 10, 10], "label": "B-item" if i == 0 else "I-item"} for i in range(1200)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.jsonl"
            path.write_text(json.dumps(sample) + '\n', encoding="utf-8")
            dataset = TokenDataset(path, tokenizer, labels)
            features = [dataset[i] for i in range(len(dataset))]
            self.assertGreater(len(features), 1)
            self.assertEqual(sum(x != -100 for row in features for x in row["labels"]), 1200)
            batch = TokenCollator(tokenizer)(features)
            self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
            pred = [[max(token, 0) for token in row["labels"]] for row in features]
            result = decode_predictions(pred, dataset.samples, tokenizer, {v: k for k, v in labels.items()})
            self.assertEqual(result, [{"item": [" ".join(["tea"] * 1200)]}])

    def test_last_accumulation_group_uses_actual_size(self):
        for name in ("roberta", "bros", "pick", "layoutlmv3", "donut", "qwen_vl"):
            with self.subTest(model=name):
                trainer = importlib.import_module(f"models.{name}.train")
                model = ToyModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                keys = ("images", "boxes", "input_ids", "attention_mask", "box_mask", "relations", "labels") if name == "pick" else ("input_ids", "attention_mask", "labels", "bbox", "pixel_values")
                if name == "donut":
                    keys = ("pixel_values", "decoder_input_ids", "decoder_attention_mask", "labels")
                elif name == "qwen_vl":
                    keys = ("pixel_values", "input_ids", "attention_mask", "image_grid_thw", "labels")
                loader = [{key: torch.zeros(1) for key in keys} for _ in range(3)]
                trainer.train_epoch(model, loader, optimizer, scheduler,
                                    torch.amp.GradScaler("cpu", enabled=False), torch.device("cpu"),
                                    False, 1, 1, 2)
                self.assertAlmostEqual(model.weight.item(), 0.32, places=5)

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = ToyModel()
            loader = DataLoader([{"input_ids": torch.zeros(1)} for _ in range(3)], batch_size=1)
            evaluate = lambda: (0.1, {"precision": 0.5, "recall": 0.5, "f1": 0.5})
            result = fit(model, loader, evaluate, root / "ckpt", root / "out",
                         epochs=1, accum_steps=2, min_epochs=1)
            self.assertEqual(result["optimizer_updates"], 2)
            restored = ToyModel()
            load_weights(restored, root / "ckpt/best_weights.pt", torch.device("cpu"))
            self.assertTrue(torch.equal(restored.weight, model.weight))
            with self.assertRaises(FileExistsError):
                fit(model, loader, evaluate, root / "ckpt", root / "out")


if __name__ == "__main__":
    unittest.main()
