import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from tools.common import resolve_data_dir


DEFAULT_CHARS = (
    " 0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


def get_char_vocab(train_path: Path | None = None) -> dict[str, int]:
    char2id = {"<pad>": 0, "<unk>": 1}
    for char in DEFAULT_CHARS:
        if char not in char2id:
            char2id[char] = len(char2id)

    if train_path is not None and train_path.exists():
        with open(train_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                for item in data.get("ocr", []):
                    for char in item.get("text", ""):
                        if char not in char2id:
                            char2id[char] = len(char2id)

    return char2id


def get_tag_vocab(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_tags = ["O"] + [lbl for lbl in labels if lbl != "O"]
    tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2tag = {idx: tag for idx, tag in enumerate(unique_tags)}
    return tag2id, id2tag


def calculate_box_center(item: dict) -> tuple[float, float]:
    quad = item["quad"]
    cx = (quad[0] + quad[2] + quad[4] + quad[6]) / 4.0
    cy = (quad[1] + quad[3] + quad[5] + quad[7]) / 4.0
    return cy, cx


def sort_reading_order(ocr_items: list[dict]) -> list[dict]:
    return sorted(ocr_items, key=calculate_box_center)


def compute_relations(boxes: list[list[float]], texts: list[str], img_w: int, img_h: int) -> np.ndarray:
    num_boxes = len(boxes)
    if num_boxes == 0:
        return np.zeros((0, 0, 6), dtype=np.float32)

    boxes_arr = np.array(boxes, dtype=np.float32)
    widths = np.maximum(boxes_arr[:, 2] - boxes_arr[:, 0], 1.0)
    heights = np.maximum(boxes_arr[:, 3] - boxes_arr[:, 1], 1.0)
    centers_x = boxes_arr[:, 0] + widths / 2.0
    centers_y = boxes_arr[:, 1] + heights / 2.0
    t_lens = np.maximum([len(t) for t in texts], 1.0)

    dx = np.abs(centers_x[:, None] - centers_x[None, :]) / max(img_w, 1.0)
    dy = np.abs(centers_y[:, None] - centers_y[None, :]) / max(img_h, 1.0)
    aspect_i = (widths / heights)[:, None] * np.ones((1, num_boxes), dtype=np.float32)
    ratio_h = heights[None, :] / heights[:, None]
    ratio_w = widths[None, :] / heights[:, None]
    ratio_t = t_lens[None, :] / t_lens[:, None]

    relations = np.stack([dx, dy, aspect_i, ratio_h, ratio_w, ratio_t], axis=-1)
    return relations.astype(np.float32)


class PickDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        dataset_name: str,
        char2id: dict[str, int],
        tag2id: dict[str, int],
        max_boxes: int = 256,
        max_text_len: int = 64,
        img_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.dataset_name = dataset_name
        self.char2id = char2id
        self.tag2id = tag2id
        self.max_boxes = max_boxes
        self.max_text_len = max_text_len
        self.img_size = img_size
        self.data_dir = resolve_data_dir(dataset_name)

        with open(jsonl_path, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img_path = self.data_dir / sample["image_path"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        target_w, target_h = self.img_size
        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        img_np = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image_tensor = torch.from_numpy((img_np - mean) / std)

        scale_x = target_w / max(orig_w, 1.0)
        scale_y = target_h / max(orig_h, 1.0)

        raw_items = sample.get("ocr", [])
        sorted_items = sort_reading_order(raw_items)[:self.max_boxes]

        boxes_list = []
        texts_list = []
        input_ids_list = []
        lengths_list = []
        labels_list = []

        for item in sorted_items:
            quad = item["quad"]
            xs = [quad[0], quad[2], quad[4], quad[6]]
            ys = [quad[1], quad[3], quad[5], quad[7]]
            x1 = max(0.0, min(float(target_w), min(xs) * scale_x))
            y1 = max(0.0, min(float(target_h), min(ys) * scale_y))
            x2 = max(0.0, min(float(target_w), max(xs) * scale_x))
            y2 = max(0.0, min(float(target_h), max(ys) * scale_y))

            if x2 <= x1:
                x2 = min(float(target_w), x1 + 1.0)
            if y2 <= y1:
                y2 = min(float(target_h), y1 + 1.0)

            boxes_list.append([x1, y1, x2, y2])

            raw_text = item.get("text", "")
            texts_list.append(raw_text)

            char_seq = [self.char2id.get(c, 1) for c in raw_text[:self.max_text_len]]
            actual_len = len(char_seq)
            lengths_list.append(actual_len)

            padded_chars = char_seq + [0] * (self.max_text_len - actual_len)
            input_ids_list.append(padded_chars)

            box_label = item.get("label", "O")
            if box_label == "O" or not box_label:
                char_labels = [0] * actual_len
            elif box_label.startswith("B-"):
                entity_name = box_label[2:]
                b_tag = self.tag2id.get(box_label, 0)
                i_tag = self.tag2id.get(f"I-{entity_name}", b_tag)
                char_labels = [b_tag] + [i_tag] * (actual_len - 1) if actual_len > 0 else []
            else:
                tag_id = self.tag2id.get(box_label, 0)
                char_labels = [tag_id] * actual_len

            padded_labels = char_labels + [0] * (self.max_text_len - actual_len)
            labels_list.append(padded_labels)

        if not boxes_list:
            boxes_list = [[0.0, 0.0, 1.0, 1.0]]
            texts_list = [""]
            input_ids_list = [[0] * self.max_text_len]
            lengths_list = [0]
            labels_list = [[0] * self.max_text_len]

        relations = compute_relations(boxes_list, texts_list, target_w, target_h)

        return {
            "image": image_tensor,
            "boxes": torch.tensor(boxes_list, dtype=torch.float32),
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "lengths": torch.tensor(lengths_list, dtype=torch.long),
            "relations": torch.from_numpy(relations),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "num_boxes": len(boxes_list),
        }


def pick_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_n = max(item["num_boxes"] for item in batch)
    max_t = batch[0]["input_ids"].shape[1]

    images = torch.stack([item["image"] for item in batch], dim=0)

    batched_boxes = torch.zeros(batch_size, max_n, 4, dtype=torch.float32)
    batched_input_ids = torch.zeros(batch_size, max_n, max_t, dtype=torch.long)
    batched_lengths = torch.zeros(batch_size, max_n, dtype=torch.long)
    batched_attention_mask = torch.ones(batch_size, max_n, max_t, dtype=torch.bool)
    batched_box_mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    batched_relations = torch.zeros(batch_size, max_n, max_n, 6, dtype=torch.float32)
    batched_labels = torch.zeros(batch_size, max_n, max_t, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["num_boxes"]
        batched_boxes[i, :n] = item["boxes"]
        batched_input_ids[i, :n] = item["input_ids"]
        batched_lengths[i, :n] = item["lengths"]
        batched_box_mask[i, :n] = True
        batched_relations[i, :n, :n] = item["relations"]
        batched_labels[i, :n] = item["labels"]

        for b_idx in range(n):
            actual_len = int(item["lengths"][b_idx].item())
            if actual_len > 0:
                batched_attention_mask[i, b_idx, :actual_len] = False

    return {
        "images": images,
        "boxes": batched_boxes,
        "input_ids": batched_input_ids,
        "lengths": batched_lengths,
        "attention_mask": batched_attention_mask,
        "box_mask": batched_box_mask,
        "relations": batched_relations,
        "labels": batched_labels,
    }


def resolve_box_tag(char_tags: list[str]) -> str:
    for tag in char_tags:
        if tag.startswith("B-"):
            return tag
    for tag in char_tags:
        if tag.startswith("I-"):
            return tag
    return "O"


def extract_entities(ocr_items: list[dict], box_tags: list[str]) -> dict[str, list[str]]:
    sample_pred: dict[str, list[str]] = {}
    current_label = None
    current_tokens: list[str] = []

    for item, tag in zip(ocr_items, box_tags):
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
    preds: list[list[list[int]]],
    samples: list[dict],
    id2tag: dict[int, str],
    max_boxes: int = 256,
) -> list[dict]:
    if len(preds) != len(samples):
        raise ValueError(f"preds ({len(preds)}) and samples ({len(samples)}) length mismatch.")

    return [
        extract_entities(
            sorted_items := sort_reading_order(sample.get("ocr", []))[:max_boxes],
            [
                resolve_box_tag([
                    id2tag.get(tid, "O")
                    for tid in (sample_preds[b_idx] if b_idx < len(sample_preds) else [])
                ])
                for b_idx in range(len(sorted_items))
            ],
        )
        for sample_preds, sample in zip(preds, samples)
    ]