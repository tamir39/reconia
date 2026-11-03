import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from torch.utils.data import DataLoader
from transformers import (
    Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2VLVideoProcessor, Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
)

from models.qwen_vl import train as trainer
from models.qwen_vl.dataset import QwenDataCollator, QwenDataset


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.weights_dir = self.root / "weights/qwen_vl"
        self.weights_dir.mkdir(parents=True)
        special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>",
                   "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
        vocab = {token: index for index, token in enumerate(special + sorted(ByteLevel.alphabet()))}
        (self.weights_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (self.weights_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        tokenizer = Qwen2TokenizerFast(
            vocab_file=str(self.weights_dir / "vocab.json"),
            merges_file=str(self.weights_dir / "merges.txt"),
            eos_token="<|im_end|>", pad_token="<|endoftext|>", additional_special_tokens=special,
        )
        template = (
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}"
            "{% for content in message['content'] %}{% if content['type'] == 'image' %}"
            "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
            "{% else %}{{ content['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        self.processor = Qwen2_5_VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                min_pixels=16, max_pixels=16, patch_size=2, temporal_patch_size=2, merge_size=2),
            video_processor=Qwen2VLVideoProcessor(),
            tokenizer=tokenizer, chat_template=template,
        )
        self.processor.save_pretrained(self.weights_dir)
        config = Qwen2_5_VLConfig(
            text_config=dict(
                vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=512, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            ),
            vision_config=dict(
                depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                out_hidden_size=32, window_size=8, fullatt_block_indexes=[0],
            ),
            image_token_id=vocab["<|image_pad|>"], video_token_id=vocab["<|video_pad|>"],
            vision_start_token_id=vocab["<|vision_start|>"], vision_end_token_id=vocab["<|vision_end|>"],
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        Qwen2_5_VLForConditionalGeneration(config).save_pretrained(self.weights_dir)
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()
        Image.new("RGB", (4, 4), "white").save(self.root / "receipt.png")
        self.gt = {"item": ["tea", "tea"]}
        self.sample = {"id": "receipt", "image_path": "receipt.png", "gt": self.gt}
        for split in ("train", "val", "test"):
            (self.cache_dir / f"{split}.jsonl").write_text(json.dumps(self.sample) + "\n", encoding="utf-8")

    def dataset(self, training=True, **kwargs):
        return QwenDataset(self.cache_dir / "val.jsonl", self.processor, self.root,
                           fields=["item"], training=training, max_new_tokens=40, **kwargs)

    def build_model(self):
        model, processor = trainer.build_model(self.weights_dir, torch.device("cpu"))
        processor = self.processor
        return model, processor

    def test_prompt_mask_padding_and_budget_without_target_leakage(self):
        inference = self.dataset(False)[0]
        feature = self.dataset()[0]
        prefix_width = len(inference["input_ids"])
        self.assertEqual(feature["input_ids"][:prefix_width], inference["input_ids"])
        self.assertEqual(feature["labels"][:prefix_width], [-100] * prefix_width)
        answer = self.processor.tokenizer(json.dumps(self.gt, sort_keys=True, separators=(",", ":")),
                                          add_special_tokens=False)["input_ids"]
        self.assertEqual(feature["labels"][prefix_width:], answer + [self.processor.tokenizer.eos_token_id])
        self.assertNotIn("tea", self.dataset().prefix)
        short = dict(feature, input_ids=feature["input_ids"][:-2],
                     attention_mask=feature["attention_mask"][:-2], labels=feature["labels"][:-2])
        batch = QwenDataCollator(self.processor)([feature, short])
        self.assertEqual(batch["labels"][1, -2:].tolist(), [-100, -100])
        short = dict(inference, input_ids=inference["input_ids"][:-2], attention_mask=inference["attention_mask"][:-2])
        batch = QwenDataCollator(self.processor, False)([inference, short])
        self.assertEqual(batch["attention_mask"][1, :2].tolist(), [0, 0])
        self.assertEqual(batch["image_grid_thw"].shape[0], 2)
        with self.assertRaisesRegex(ValueError, "context budget"):
            self.dataset(max_length=10)[0]
        with self.assertRaisesRegex(ValueError, "target too long"):
            QwenDataset(self.cache_dir / "val.jsonl", self.processor, self.root,
                        fields=["item"], max_new_tokens=2)[0]
        (self.cache_dir / "val.jsonl").write_text(json.dumps({
            "id": "receipt", "image_path": "receipt.png"}) + "\n", encoding="utf-8")
        self.assertNotIn("labels", self.dataset(False)[0])

    def test_lora_freezes_base_and_evaluation_only_scores_continuation(self):
        model, processor = self.build_model()
        self.assertTrue(all("lora_" in name and "visual" not in name
                            for name, param in model.named_parameters() if param.requires_grad))
        self.assertTrue(any(param.requires_grad for param in model.parameters()))
        loader = DataLoader(self.dataset(), collate_fn=QwenDataCollator(processor))
        inference_loader = DataLoader(self.dataset(False), collate_fn=QwenDataCollator(processor, False))
        for text, expected_f1 in [(json.dumps(self.gt), 1.0), ('{"item":[', 0.0)]:
            def generate(**kwargs):
                self.assertNotIn("labels", kwargs)
                self.assertEqual(kwargs["input_ids"].tolist(), [self.dataset(False)[0]["input_ids"]])
                answer = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                return torch.cat([kwargs["input_ids"], torch.tensor([answer])], dim=1)
            with patch.object(model, "generate", side_effect=generate):
                loss, metrics = trainer.evaluate_epoch(
                    model, loader, inference_loader, loader.dataset.samples, processor,
                    torch.device("cpu"), "[Val]", max_new_tokens=40)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertEqual(metrics["f1"], expected_f1)

    def test_real_lora_train_resume_and_adapter_only_test_checkpoint(self):
        original_build = trainer.build_model
        first_model = []
        initial_state = {}

        def build_model(weights_dir, device):
            model, processor = original_build(weights_dir, device)
            processor = self.processor
            if not first_model:
                first_model.append(model)
                initial_state.update({name: param.detach().clone() for name, param in model.named_parameters()})
            return model, processor

        with (
            patch.object(trainer, "verify_baseline_data"),
            patch.object(trainer, "get_repo_dir", return_value=self.root),
            patch.object(trainer, "get_cache_dir", return_value=self.cache_dir),
            patch.object(trainer, "get_data_dir", return_value=self.root),
            patch.object(trainer, "load_labels", return_value=["O", "B-item", "I-item"]),
            patch.object(trainer, "build_model", side_effect=build_model),
        ):
            trainer.train_model("sroie", epochs=1, min_epochs=1, max_new_tokens=40)
            parameters = dict(first_model[0].named_parameters())
            self.assertTrue(any(not torch.equal(initial_state[name], param)
                                for name, param in parameters.items() if param.requires_grad))
            self.assertTrue(all(torch.equal(initial_state[name], param)
                                for name, param in parameters.items() if not param.requires_grad))
            checkpoint_dir = self.root / "checkpoints/sroie/qwen_vl"
            out_dir = self.root / "results/sroie/qwen_vl"
            self.assertTrue((out_dir / "history.csv").exists())
            self.assertFalse((out_dir / "test_metrics.json").exists())
            state = torch.load(checkpoint_dir / "best_weights.pt", weights_only=True)
            self.assertTrue(state and all("lora_" in key for key in state))
            self.assertTrue((checkpoint_dir / "adapter_config.json").exists())
            with self.assertRaises(FileExistsError):
                trainer.train_model("sroie", epochs=1, max_new_tokens=40)
            trainer.train_model("sroie", epochs=2, min_epochs=1, resume=True, max_new_tokens=40)
            checkpoint = torch.load(checkpoint_dir / "last_checkpoint.pt", weights_only=True)
            self.assertEqual([row["epoch"] for row in checkpoint["history"]], [1, 2])
            best_before = (checkpoint_dir / "best_weights.pt").read_bytes()
            trainer.train_model("sroie", test_only=True, max_new_tokens=40)
            result = json.loads((out_dir / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["model"], "qwen_vl")
            self.assertTrue(all(0 <= result[key] <= 1 for key in ("precision", "recall", "f1")))
            self.assertEqual(best_before, (checkpoint_dir / "best_weights.pt").read_bytes())


if __name__ == "__main__":
    unittest.main()
