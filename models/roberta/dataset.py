import json
from pathlib import Path

from torch.utils.data import Dataset


def get_labels(cache_dir: Path | str) -> list[str]:
    cache_path = Path(cache_dir)
    labels = set()
    for split_file in ("train.jsonl", "val.jsonl"):
        split_path = cache_path / split_file
        with open(split_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                for item in data.get("ocr", []):
                    label = item.get("label")
                    if label:
                        labels.add(label)

    return sorted(labels)


class RobertaDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path | str,
        tokenizer,
        label2id: dict[str, int],
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        sample = self.samples[idx]
        ocr_items = sample.get("ocr", [])

        words = [item.get("text", "") for item in ocr_items]
        box_labels = [item.get("label", "O") for item in ocr_items]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            add_prefix_space=True,
        )

        box_ids = encoding.word_ids()
        labels = []
        prev_box_idx = None

        default_label_id = self.label2id.get("O", 0)
        for box_idx in box_ids:
            if box_idx is None:
                labels.append(-100)
            elif box_idx != prev_box_idx:
                label_name = box_labels[box_idx]
                labels.append(self.label2id.get(label_name, default_label_id))
            else:
                labels.append(-100)
            prev_box_idx = box_idx

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def align_box_labels(
    pred_ids: list[int],
    words: list[str],
    tokenizer,
    id2label: dict[int, str],
    max_length: int = 512,
) -> dict[int, str]:
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        add_prefix_space=True,
    )

    box_labels: dict[int, str] = {}
    prev_box_idx = None

    for tok_idx, box_idx in enumerate(encoding.word_ids()):
        if box_idx is None:
            continue
        if box_idx != prev_box_idx:
            if tok_idx < len(pred_ids):
                pred_label_id = pred_ids[tok_idx]
                box_labels[box_idx] = id2label.get(pred_label_id, "O")
            prev_box_idx = box_idx

    return box_labels


def extract_entities(ocr_items: list[dict], box_labels: dict[int, str]) -> dict[str, list[str]]:
    sample_pred: dict[str, list[str]] = {}
    current_label = None
    current_tokens: list[str] = []

    for box_idx, item in enumerate(ocr_items):
        tag = box_labels.get(box_idx, "O")
        text = item.get("text", "").strip()
        if not text:
            continue

        if tag.startswith("B-"):
            if current_label and current_tokens:
                sample_pred.setdefault(current_label, []).append(" ".join(current_tokens))
            current_label = tag[2:]
            current_tokens = [text]
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_label == entity_type:
                current_tokens.append(text)
            else:
                if current_label and current_tokens:
                    sample_pred.setdefault(current_label, []).append(" ".join(current_tokens))
                current_label = entity_type
                current_tokens = [text]
        else:
            if current_label and current_tokens:
                sample_pred.setdefault(current_label, []).append(" ".join(current_tokens))
                current_label = None
                current_tokens = []

    if current_label and current_tokens:
        sample_pred.setdefault(current_label, []).append(" ".join(current_tokens))

    return sample_pred


def reconstruct_predictions(
    preds: list[list[int]],
    samples: list[dict],
    tokenizer,
    id2label: dict[int, str],
    max_length: int = 512,
) -> list[dict]:
    reconstructed = []

    for pred_ids, sample in zip(preds, samples):
        ocr_items = sample.get("ocr", [])
        words = [item.get("text", "") for item in ocr_items]
        box_labels = align_box_labels(pred_ids, words, tokenizer, id2label, max_length)

        sample_pred = extract_entities(ocr_items, box_labels)
        reconstructed.append(sample_pred)

    return reconstructed
