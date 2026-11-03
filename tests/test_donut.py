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
    DonutImageProcessor, DonutProcessor, DonutSwinConfig, DonutSwinModel,
    MBartConfig, MBartForCausalLM, RobertaTokenizerFast, VisionEncoderDecoderModel,
)

from models.donut import train as trainer
from models.donut.dataset import DonutDataCollator, DonutDataset


class DonutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.weights_dir = self.root / "weights/donut"
        self.weights_dir.mkdir(parents=True)
        tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + sorted(ByteLevel.alphabet())
        (self.weights_dir / "vocab.json").write_text(
            json.dumps({token: index for index, token in enumerate(tokens)}), encoding="utf-8")
        (self.weights_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        tokenizer = RobertaTokenizerFast(
            vocab_file=str(self.weights_dir / "vocab.json"),
            merges_file=str(self.weights_dir / "merges.txt"),
        )
        self.processor = DonutProcessor(
            DonutImageProcessor(size={"height": 32, "width": 32}), tokenizer)
        self.processor.save_pretrained(self.weights_dir)
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()
        Image.new("RGB", (40, 50), "white").save(self.root / "receipt.png")
        self.gt = {"item": ["tea", "tea"]}
        self.sample = {"id": "receipt", "image_path": "receipt.png", "gt": self.gt}
        for split in ("train", "val", "test"):
            (self.cache_dir / f"{split}.jsonl").write_text(json.dumps(self.sample) + "\n", encoding="utf-8")
        encoder = DonutSwinModel(DonutSwinConfig(
            image_size=32, patch_size=4, embed_dim=16, depths=[1], num_heads=[2], window_size=4))
        decoder = MBartForCausalLM(MBartConfig(
            vocab_size=len(tokenizer), d_model=16, decoder_layers=1, decoder_attention_heads=2,
            encoder_layers=1, encoder_attention_heads=2, decoder_ffn_dim=32, encoder_ffn_dim=32,
            max_position_embeddings=64, is_decoder=True, add_cross_attention=True,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ))
        VisionEncoderDecoderModel(encoder=encoder, decoder=decoder).save_pretrained(self.weights_dir)

    def dataset(self, training=True, max_new_tokens=40):
        return DonutDataset(self.cache_dir / "val.jsonl", self.processor, self.root,
                            training=training, max_new_tokens=max_new_tokens)

    def build_model(self, weights_dir, device):
        model, processor = trainer.build_model(weights_dir, device)
        processor.image_processor.size = {"height": 32, "width": 32}
        return model, processor

    def test_target_shift_padding_and_inference_without_gt_or_ocr(self):
        feature = self.dataset()[0]
        tokenizer = self.processor.tokenizer
        expected = tokenizer(json.dumps(self.gt, sort_keys=True, separators=(",", ":")),
                             add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        self.assertEqual(feature["labels"], expected)
        self.assertEqual(feature["decoder_input_ids"], [tokenizer.bos_token_id] + expected[:-1])
        shorter = dict(feature, labels=expected[:3], decoder_input_ids=feature["decoder_input_ids"][:3])
        batch = DonutDataCollator(self.processor)([feature, shorter])
        self.assertTrue(torch.all(batch["labels"][1, 3:] == -100))
        self.assertTrue(torch.all(batch["decoder_attention_mask"][1, 3:] == 0))
        # Inference must work even when neither GT nor annotation text/boxes exist.
        (self.cache_dir / "val.jsonl").write_text(json.dumps({
            "id": "receipt", "image_path": "receipt.png"}) + "\n", encoding="utf-8")
        inference = self.dataset(training=False)[0]
        self.assertEqual(set(inference), {"pixel_values", "decoder_input_ids"})
        self.assertEqual(inference["decoder_input_ids"], [tokenizer.bos_token_id])
        (self.cache_dir / "val.jsonl").write_text(json.dumps(self.sample) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "target too long"):
            self.dataset(max_new_tokens=2)[0]

    def test_evaluation_scores_generated_json_and_keeps_duplicates(self):
        model, processor = self.build_model(self.weights_dir, torch.device("cpu"))
        dataset = self.dataset()
        loader = DataLoader(dataset, collate_fn=DonutDataCollator(processor))
        inference_loader = DataLoader(self.dataset(False), collate_fn=DonutDataCollator(processor, False))
        for text, expected_f1 in [(json.dumps(self.gt), 1.0), ('{"item":[', 0.0)]:
            def generate(**kwargs):
                self.assertNotIn("labels", kwargs)
                self.assertEqual(kwargs["decoder_input_ids"].tolist(), [[processor.tokenizer.bos_token_id]])
                ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                return torch.tensor([[processor.tokenizer.bos_token_id] + ids + [processor.tokenizer.eos_token_id]])
            with patch.object(model, "generate", side_effect=generate):
                loss, metrics = trainer.evaluate_epoch(
                    model, loader, inference_loader, dataset.samples, processor,
                    torch.device("cpu"), "[Val]", max_new_tokens=40)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertEqual(metrics["f1"], expected_f1)

    def test_real_train_generation_resume_and_separate_test(self):
        original_build = trainer.build_model

        def build_model(weights_dir, device):
            model, processor = original_build(weights_dir, device)
            processor.image_processor.size = {"height": 32, "width": 32}
            return model, processor

        with (
            patch.object(trainer, "verify_baseline_data"),
            patch.object(trainer, "get_repo_dir", return_value=self.root),
            patch.object(trainer, "get_cache_dir", return_value=self.cache_dir),
            patch.object(trainer, "get_data_dir", return_value=self.root),
            patch.object(trainer, "build_model", side_effect=build_model),
        ):
            trainer.train_model("sroie", epochs=1, min_epochs=1, max_new_tokens=40)
            checkpoint_dir = self.root / "checkpoints/sroie/donut"
            out_dir = self.root / "results/sroie/donut"
            self.assertTrue((out_dir / "history.csv").exists())
            self.assertFalse((out_dir / "test_metrics.json").exists())
            with self.assertRaises(FileExistsError):
                trainer.train_model("sroie", epochs=1, max_new_tokens=40)
            trainer.train_model("sroie", epochs=2, min_epochs=1, resume=True, max_new_tokens=40)
            checkpoint = torch.load(checkpoint_dir / "last_checkpoint.pt", weights_only=True)
            self.assertEqual([row["epoch"] for row in checkpoint["history"]], [1, 2])
            best_before = (checkpoint_dir / "best_weights.pt").read_bytes()
            trainer.train_model("sroie", test_only=True, max_new_tokens=40)
            result = json.loads((out_dir / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["model"], "donut")
            self.assertTrue(all(0 <= result[key] <= 1 for key in ("precision", "recall", "f1")))
            self.assertEqual(best_before, (checkpoint_dir / "best_weights.pt").read_bytes())


if __name__ == "__main__":
    unittest.main()
