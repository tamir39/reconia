import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


def encode_document(sample, tokenizer, max_length=512):
    options = ({"boxes": [item["box"] for item in sample["ocr"]]}
               if hasattr(tokenizer, "pad_token_box") else {"is_split_into_words": True})
    encoding = tokenizer(
        [item["text"] for item in sample["ocr"]], **options,
        max_length=max_length, truncation=True, stride=64,
        return_overflowing_tokens=True,
    )
    seen, windows = set(), []
    for index, ids in enumerate(encoding["input_ids"]):
        word_ids = encoding.word_ids(index)
        owners = []
        for word in word_ids:
            owners.append(word is not None and word not in seen)
            if word is not None:
                seen.add(word)
        if any(owners):
            windows.append((ids, encoding["attention_mask"][index], word_ids, owners))
    if not windows:
        raise ValueError(f"Empty annotation text: {sample['id']}")
    return windows


class TokenDataset(Dataset):
    def __init__(self, data_file, tokenizer, label2id, max_length=512,
                 layout=None, image_processor=None, data_dir=None):
        self.samples = [json.loads(line) for line in Path(data_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer, self.label2id = tokenizer, label2id
        self.layout, self.image_processor = layout, image_processor
        self.data_dir = Path(data_dir) if data_dir else None
        self.windows = [(index, window) for index, sample in enumerate(self.samples)
                        for window in encode_document(sample, tokenizer, max_length)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sample_index, (ids, mask, words, owners) = self.windows[index]
        sample = self.samples[sample_index]
        items = sample["ocr"]
        feature = {
            "input_ids": ids, "attention_mask": mask,
            "labels": [self.label2id[items[word]["label"]] if owner else -100
                       for word, owner in zip(words, owners)],
        }
        if self.layout:
            divisor = 1000.0 if self.layout == "bros" else 1
            feature["bbox"] = [[coord / divisor for coord in items[word]["box"]]
                               if word is not None else [0, 0, 0, 0] for word in words]
        if self.image_processor is not None:
            with Image.open(self.data_dir / sample["image_path"]) as image:
                feature["pixel_values"] = self.image_processor(
                    image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return feature


class TokenCollator:
    def __init__(self, tokenizer, layout=None):
        self.pad_token_id, self.layout = tokenizer.pad_token_id, layout

    def __call__(self, features):
        width = max(len(item["input_ids"]) for item in features)
        batch = {}
        for key, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key] = torch.tensor([item[key] + [fill] * (width - len(item[key]))
                                       for item in features], dtype=torch.long)
        if "bbox" in features[0]:
            batch["bbox"] = torch.tensor([
                item["bbox"] + [[0, 0, 0, 0]] * (width - len(item["bbox"])) for item in features
            ], dtype=torch.float32 if self.layout == "bros" else torch.long)
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([item["pixel_values"] for item in features])
        return batch


def extract_entities(items, labels):
    entities, texts, current = {}, [], None
    for index, item in enumerate(items):
        label = labels.get(index, "O")
        if label.startswith("I-") and label[2:] == current:
            texts.append(item["text"].strip())
            continue
        if current and texts:
            entities.setdefault(current, []).append(" ".join(texts))
        current = label[2:] if label.startswith(("B-", "I-")) else None
        texts = [item["text"].strip()] if current else []
    if current and texts:
        entities.setdefault(current, []).append(" ".join(texts))
    return entities


def decode_predictions(all_pred_ids, samples, tokenizer, id2label, max_length=512):
    predictions, cursor = [], 0
    for sample in samples:
        labels = {}
        for ids, _, words, owners in encode_document(sample, tokenizer, max_length):
            if cursor >= len(all_pred_ids) or len(all_pred_ids[cursor]) < len(ids):
                raise ValueError("Missing window predictions")
            for token, word, owner in zip(all_pred_ids[cursor], words, owners):
                if owner:
                    labels[word] = id2label[token]
            cursor += 1
        predictions.append(extract_entities(sample["ocr"], labels))
    if cursor != len(all_pred_ids):
        raise ValueError("Unexpected extra window predictions")
    return predictions
