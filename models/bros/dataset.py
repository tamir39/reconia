import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class BrosDataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        with open(data_file, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list]:
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

        bbox: list[list[float]] = []
        labels: list[int] = []
        bbox_first_token_mask: list[bool] = []
        prev_word_idx = None

        for word_idx in encoding.word_ids():
            if word_idx is None:
                bbox.append([0.0, 0.0, 0.0, 0.0])
                labels.append(-100)
                bbox_first_token_mask.append(False)
            else:
                bbox.append([max(0.0, min(1.0, coord / 1000.0)) for coord in boxes[word_idx]])
                if word_idx != prev_word_idx:
                    labels.append(self.label2id[word_labels[word_idx]])
                    bbox_first_token_mask.append(True)
                else:
                    labels.append(-100)
                    bbox_first_token_mask.append(False)
                prev_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"],
            "bbox": bbox,
            "attention_mask": encoding["attention_mask"],
            "bbox_first_token_mask": bbox_first_token_mask,
            "labels": labels,
        }


class BrosDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids: list[list[int]] = []
        bbox: list[list[list[float]]] = []
        attention_mask: list[list[int]] = []
        bbox_first_token_mask: list[list[bool]] = []
        labels: list[list[int]] = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            bbox.append(item["bbox"] + [[0.0, 0.0, 0.0, 0.0]] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            bbox_first_token_mask.append(item.get("bbox_first_token_mask", []) + [False] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bbox_first_token_mask": torch.tensor(bbox_first_token_mask, dtype=torch.bool),
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
    max_length: int = 512,
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

    bbox: list[list[float]] = []
    bbox_first_token_mask: list[bool] = []
    prev_word_idx = None

    for word_idx in encoding.word_ids():
        if word_idx is None:
            bbox.append([0.0, 0.0, 0.0, 0.0])
            bbox_first_token_mask.append(False)
        else:
            bbox.append([max(0.0, min(1.0, coord / 1000.0)) for coord in boxes[word_idx]])
            if word_idx != prev_word_idx:
                bbox_first_token_mask.append(True)
            else:
                bbox_first_token_mask.append(False)
            prev_word_idx = word_idx

    bbox = torch.tensor([bbox], dtype=torch.float, device=device)
    bbox_first_token_mask = torch.tensor([bbox_first_token_mask], dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            bbox=bbox,
            attention_mask=encoding["attention_mask"].to(device),
            bbox_first_token_mask=bbox_first_token_mask,
        )

    pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    labels = align_labels(pred_ids, encoding.word_ids(), id2label)
    return extract_entities(ocr, labels)
