import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class LayoutLMv3Dataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        data_dir: Path | None = None,
        max_length: int = 512,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.data_dir = data_dir
        self.max_length = max_length
        self.image_size = image_size

        with open(data_file, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, sample: dict) -> torch.Tensor:
        if self.data_dir and "image_path" in sample:
            image_path = self.data_dir / sample["image_path"]
            if image_path.exists():
                with Image.open(image_path) as img:
                    img = img.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    arr = (arr - 0.5) / 0.5
                    return torch.from_numpy(arr.transpose(2, 0, 1))

        return torch.zeros((3, *self.image_size), dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list]:
        sample = self.samples[idx]
        ocr = sample["ocr"]
        words = [item["text"] for item in ocr]
        boxes = [item["box"] for item in ocr]
        word_labels = [item["label"] for item in ocr]

        encoding = self.tokenizer(
            words,
            max_length=self.max_length,
            is_split_into_words=True,
            truncation=True,
        )

        bbox: list[list[int]] = []
        labels: list[int] = []
        prev_word_idx = None

        for word_idx in encoding.word_ids():
            if word_idx is None:
                bbox.append([0, 0, 1000, 1000])
                labels.append(-100)
            else:
                box = boxes[word_idx]
                clamped_box = [
                    max(0, min(1000, int(box[0]))),
                    max(0, min(1000, int(box[1]))),
                    max(0, min(1000, int(box[2]))),
                    max(0, min(1000, int(box[3]))),
                ]
                bbox.append(clamped_box)
                if word_idx != prev_word_idx:
                    labels.append(self.label2id[word_labels[word_idx]])
                else:
                    labels.append(-100)
                prev_word_idx = word_idx

        pixel_values = self._process_image(sample)

        return {
            "input_ids": encoding["input_ids"],
            "bbox": bbox,
            "attention_mask": encoding["attention_mask"],
            "pixel_values": pixel_values,
            "labels": labels,
        }


class LayoutLMv3DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids: list[list[int]] = []
        bbox: list[list[list[int]]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        pixel_values: list[torch.Tensor] = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            bbox.append(item["bbox"] + [[0, 0, 0, 0]] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            pixel_values.append(item["pixel_values"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": torch.stack(pixel_values, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def align_labels(
    pred_ids: list[int],
    word_ids: list[int | None],
    id2label: dict[int, str],
) -> dict[int, str]:
    labels: dict[int, str] = {}
    prev_word_idx = None

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != prev_word_idx:
            labels[word_idx] = id2label[pred_ids[token_idx]]
            prev_word_idx = word_idx

    return labels


def extract_entities(
    ocr: list[dict],
    labels: dict[int, str],
) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    current_entity = None
    current_texts: list[str] = []

    for word_idx, item in enumerate(ocr):
        label = labels.get(word_idx, "O")
        text = item["text"].strip()
        if not text:
            continue

        if label.startswith("I-") and label[2:] == current_entity:
            current_texts.append(text)
        else:
            if current_entity and current_texts:
                entities.setdefault(current_entity, []).append(" ".join(current_texts))

            if label.startswith(("B-", "I-")):
                current_entity = label[2:]
                current_texts = [text]
            else:
                current_entity = None
                current_texts = []

    if current_entity and current_texts:
        entities.setdefault(current_entity, []).append(" ".join(current_texts))

    return entities


def decode_predictions(
    all_pred_ids: list[list[int]],
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict[int, str],
    max_length: int = 512,
) -> list[dict[str, list[str]]]:
    preds: list[dict[str, list[str]]] = []

    for pred_ids, sample in zip(all_pred_ids, samples):
        ocr = sample["ocr"]
        words = [item["text"] for item in ocr]
        encoding = tokenizer(
            words,
            max_length=max_length,
            is_split_into_words=True,
            truncation=True,
        )
        labels = align_labels(pred_ids, encoding.word_ids(), id2label)
        preds.append(extract_entities(ocr, labels))

    return preds


def predict_sample(
    ocr: list[dict],
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict[int, str],
    device: torch.device,
    image: Image.Image | None = None,
    max_length: int = 512,
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, list[str]]:
    words = [item["text"] for item in ocr]
    boxes = [item["box"] for item in ocr]

    encoding = tokenizer(
        words,
        max_length=max_length,
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt",
    )

    bbox: list[list[int]] = []
    for word_idx in encoding.word_ids():
        if word_idx is None:
            bbox.append([0, 0, 1000, 1000])
        else:
            box = boxes[word_idx]
            bbox.append([
                max(0, min(1000, int(box[0]))),
                max(0, min(1000, int(box[1]))),
                max(0, min(1000, int(box[2]))),
                max(0, min(1000, int(box[3]))),
            ])

    bbox_tensor = torch.tensor([bbox], dtype=torch.long, device=device)

    if image is not None:
        img_resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        pixel_values = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    else:
        pixel_values = torch.zeros((1, 3, *image_size), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            bbox=bbox_tensor,
            attention_mask=encoding["attention_mask"].to(device),
            pixel_values=pixel_values,
        )

    pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    labels = align_labels(pred_ids, encoding.word_ids(), id2label)
    return extract_entities(ocr, labels)