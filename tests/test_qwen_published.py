import ast
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from demo.assets import ROOT
from models.qwen_vl.dataset import (
    QwenVLDataCollator, QwenVLDataset, SYSTEM_PROMPT, extract_entities, format_prompt, predict_sample,
)
from tools.verify_final import verify
from utils.metrics import evaluate_dataset


class Encoding(dict):
    def to(self, device):
        return Encoding({key: value.to(device) for key, value in self.items()})


class Processor:
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [4, 5], pad_token_id=0)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "training" if any(message["role"] == "assistant" for message in messages) else "prompt"

    def __call__(self, images, texts, **kwargs):
        ids = [9, 4, 5, 77, 78, 2] if texts == ["training"] else [9, 4, 5]
        return Encoding(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        pixel_values=torch.zeros(1, 3), image_grid_thw=torch.ones(1, 3, dtype=torch.long))

    def batch_decode(self, tokens, **kwargs):
        self.decoded_ids = tokens.tolist()
        return ['{"total":"10"}']


class QwenPublishedTests(unittest.TestCase):
    def test_prompt_contains_fields_not_document_answers(self):
        prompt = format_prompt(["O", "B-total", "I-total", "B-company", "B-total"])
        self.assertEqual(prompt.count("total"), 1)
        self.assertIn("total, company", prompt)
        self.assertIn("valid JSON", SYSTEM_PROMPT)

    def test_training_masks_prompt_and_keeps_answer_targets(self):
        processor = Processor()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (8, 8)).save(root / "receipt.png")
            (root / "data.jsonl").write_text(json.dumps({"image_path": "receipt.png", "gt": {"total": ["10"]}}))
            item = QwenVLDataset(root, root / "data.jsonl", processor, ["B-total"])[0]
        self.assertEqual(item["prompt_len"], 3)
        self.assertEqual(item["labels"].tolist(), [-100, -100, -100, 77, 78, 2])
        self.assertEqual(json.loads(processor.messages[-1]["content"]), {"total": ["10"]})

    def test_collator_masks_padding_without_changing_prompt_lengths(self):
        def item(ids, length):
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids)),
                    "labels": torch.tensor([-100] * length + ids[length:]), "prompt_len": length,
                    "pixel_values": torch.zeros(1, 3), "image_grid_thw": torch.ones(1, 3)}
        batch = QwenVLDataCollator(Processor())([item([9, 4, 5, 77], 3), item([9, 4, 5], 2)])
        self.assertEqual(batch["input_ids"].tolist(), [[9, 4, 5, 77], [9, 4, 5, 0]])
        self.assertEqual(batch["labels"][1].tolist(), [-100, -100, 5, -100])
        self.assertEqual(batch["prompt_len"].tolist(), [3, 2])

    def test_inference_has_no_assistant_answer_and_decodes_only_new_tokens(self):
        processor = Processor()
        model = SimpleNamespace(eval=Mock(), generate=Mock(return_value=torch.tensor([[9, 4, 5, 77, 2]])))
        result = predict_sample(Image.new("RGB", (8, 8)), model, processor, torch.device("cpu"), ["B-total"])
        self.assertEqual(result, {"total": ["10"]})
        self.assertNotIn("assistant", [message["role"] for message in processor.messages])
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 1024)
        self.assertEqual(processor.decoded_ids, [[77, 2]])

    def test_evaluation_keeps_gt_out_of_prompt_but_uses_gt_length_budget(self):
        # Run the actual evaluator body without importing GPU-only PEFT/bitsandbytes.
        module = ast.parse((ROOT / "models/qwen_vl/train.py").read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_epoch")
        processor = Processor()
        model = Mock(return_value=SimpleNamespace(loss=torch.tensor(.25)))
        model.generate.return_value = torch.tensor([[0, 9, 4, 77], [8, 9, 4, 77]])
        batch = {"input_ids": torch.tensor([[9, 4, 77, 2, 0], [8, 9, 4, 77, 2]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
                 "labels": torch.tensor([[-100, -100, 77, 2, -100], [-100, -100, -100, 77, 2]]),
                 "prompt_len": torch.tensor([2, 3]), "pixel_values": torch.zeros(2, 3),
                 "image_grid_thw": torch.ones(2, 3)}
        namespace = {"torch": torch, "DataLoader": object, "AutoProcessor": object,
                     "gc": SimpleNamespace(collect=lambda: None), "tqdm": lambda loader, **kwargs: loader,
                     "decode_predictions": lambda tokens, _: [{"total": ["10"]} for row in tokens],
                     "evaluate_dataset": evaluate_dataset}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "qwen_eval", "exec"), namespace)
        loss, metrics = namespace["evaluate_epoch"](model, [batch], [{"gt": {"total": ["10"]}}] * 2,
                                                    processor, torch.device("cpu"), "test")
        generated = model.generate.call_args.kwargs
        self.assertEqual(generated["input_ids"].tolist(), [[0, 9, 4], [8, 9, 4]])
        self.assertEqual(generated["max_new_tokens"], 34)  # 2 GT answer tokens + 32, as published.
        self.assertEqual(loss, .25)
        self.assertEqual(metrics["f1"], 1)

    def test_results_retain_their_historical_output_versions(self):
        registry = json.loads((ROOT / "reports/benchmarks/additional-models.json").read_text())
        qwen = registry["models"]["qwen_vl"]
        self.assertEqual({dataset: info["output_version"] for dataset, info in qwen["datasets"].items()},
                         {"sroie": 1, "cord_v2": 3, "wildreceipt": 5})
        self.assertAlmostEqual(qwen["datasets"]["sroie"]["metrics"]["f1"], .7845543125225549)
        self.assertAlmostEqual(qwen["datasets"]["cord_v2"]["metrics"]["f1"], .8192112456071848)
        self.assertFalse(qwen["interactive_demo"])
        self.assertFalse(qwen["inference_rerun"])
        self.assertEqual(verify()["published_result_pairs"], 18)

    def test_json_parser_preserves_repeated_values_and_ignores_null(self):
        result = extract_entities('```json\n{"total":["10","10",null],"company":" A "}\n```')
        self.assertEqual(result, {"total": ["10", "10"], "company": ["A"]})
        self.assertEqual(extract_entities("not JSON"), {})


if __name__ == "__main__":
    unittest.main()
