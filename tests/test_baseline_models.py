import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from transformers import (
    BertTokenizerFast, BrosConfig, BrosForTokenClassification,
    RobertaConfig,
    RobertaForTokenClassification, RobertaTokenizerFast,
)

from models.bros import train as bros_trainer
from models.bros.dataset import BrosDataCollator, BrosDataset
from models.roberta import train as roberta_trainer
from models.roberta.dataset import RobertaDataCollator, RobertaDataset
from models.pick import model as pick_model
from models.pick import train as pick_trainer
from models.pick.dataset import PickDataset, PickDataCollator, build_char_vocab, build_tag_vocab


class BaselineModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        Image.new("RGB", (64, 64), "white").save(self.root / "receipt.png")
        self.samples = [
            {"id": "two", "image_path": "receipt.png", "ocr": [
                {"text": "tea", "box": [0, 0, 300, 200], "label": "B-item"},
                {"text": "tea", "box": [400, 0, 700, 200], "label": "I-item"}],
             "gt": {"item": ["tea tea"]}},
            {"id": "one", "image_path": "receipt.png", "ocr": [
                {"text": "tea", "box": [0, 0, 300, 200], "label": "B-item"}],
             "gt": {"item": ["tea"]}},
        ]
        self.path = self.root / "val.jsonl"
        self.path.write_text("\n".join(json.dumps(s) for s in self.samples) + "\n", encoding="utf-8")
        self.labels = {"O": 0, "B-item": 1, "I-item": 2}

    def train_and_check(self, trainer, model, loader):
        parameter = next(model.parameters())
        before = parameter.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        loss = trainer.train_epoch(
            model, loader, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False),
            torch.device("cpu"), False, 1, 1, 2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, parameter))

    def test_roberta_and_bros_real_forward_backward_and_decode(self):
        tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + sorted(ByteLevel.alphabet())
        (self.root / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
        (self.root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        roberta_tokenizer = RobertaTokenizerFast(
            vocab_file=str(self.root / "vocab.json"), merges_file=str(self.root / "merges.txt"),
            add_prefix_space=True)
        (self.root / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntea\n", encoding="utf-8")
        bros_tokenizer = BertTokenizerFast(vocab_file=str(self.root / "vocab.txt"))
        for name, trainer, tokenizer, dataset_class, collator, model in [
            ("roberta", roberta_trainer, roberta_tokenizer, RobertaDataset,
             RobertaDataCollator(roberta_tokenizer),
             RobertaForTokenClassification(RobertaConfig(
                 vocab_size=len(roberta_tokenizer), hidden_size=32, intermediate_size=48,
                 num_hidden_layers=1, num_attention_heads=4, num_labels=3))),
            ("bros", bros_trainer, bros_tokenizer, BrosDataset, BrosDataCollator(bros_tokenizer),
             BrosForTokenClassification(BrosConfig(
                 vocab_size=len(bros_tokenizer), hidden_size=64, intermediate_size=80,
                 num_hidden_layers=1, num_attention_heads=4, num_labels=3))),
        ]:
            with self.subTest(model=name):
                dataset = dataset_class(self.path, tokenizer, self.labels)
                loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
                self.train_and_check(trainer, model, loader)
                loss, metrics = trainer.evaluate_epoch(
                    model, loader, dataset.samples, tokenizer,
                    {v: k for k, v in self.labels.items()}, torch.device("cpu"), "[Val]")
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                self.assertTrue(all(0 <= metrics[key] <= 1 for key in ("precision", "recall", "f1")))

    def test_pick_real_modules_with_padding_forward_backward_and_decode(self):
        encoder = pick_model.MultimodalEncoder
        graph = pick_model.GLCN
        decoder = pick_model.BiLSTMDecoder
        with (
            patch.object(pick_model, "resnet50", side_effect=lambda **kwargs: resnet50(weights=None)),
            patch.object(pick_model, "MultimodalEncoder", side_effect=lambda **kwargs: encoder(
                **kwargs, d_model=32, num_layers=1, nhead=4, dim_feedforward=48)),
            patch.object(pick_model, "GLCN", side_effect=lambda: graph(d_model=32)),
            patch.object(pick_model, "BiLSTMDecoder", side_effect=lambda **kwargs: decoder(
                **kwargs, d_model=32, hidden_size=16, num_layers=1)),
        ):
            model = pick_model.PICKModel(vocab_size=len(build_char_vocab()), num_tags=3)
        tag2id, id2tag = build_tag_vocab(list(self.labels))
        dataset = PickDataset(self.path, self.root, build_char_vocab(), tag2id,
                              max_text_len=8, image_size=(64, 64))
        loader = DataLoader(dataset, batch_size=2, collate_fn=PickDataCollator())
        self.train_and_check(pick_trainer, model, loader)
        loss, metrics = pick_trainer.evaluate_epoch(
            model, loader, dataset.samples, id2tag, torch.device("cpu"), "[Val]")
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(all(0 <= metrics[key] <= 1 for key in ("precision", "recall", "f1")))


if __name__ == "__main__":
    unittest.main()
