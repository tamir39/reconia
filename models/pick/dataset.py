import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

DEFAULT_CHARS = (
    r"0123456789"
    r"abcdefghijklmnopqrstuvwxyz"
    r"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    r"£€¥×"
)


def build_char_vocab() -> dict[str, int]:
    ordered_chars = ["<pad>", "<unk>"] + list(dict.fromkeys(DEFAULT_CHARS))
    return {char: idx for idx, char in enumerate(ordered_chars)}


def build_tag_vocab(tags: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    ordered_tags = ["O"] + [tag for tag in dict.fromkeys(tags) if tag != "O"]
    tag2id = {tag: idx for idx, tag in enumerate(ordered_tags)}
    id2tag = {idx: tag for idx, tag in enumerate(ordered_tags)}
    return tag2id, id2tag


def compute_relations(
    boxes: list[list[float]],
    texts: list[str],
    width: int,
    height: int,
) -> np.ndarray:
    boxes = np.array(boxes, dtype=np.float32)
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    centers_x = boxes[:, 0] + widths / 2.0
    centers_y = boxes[:, 1] + heights / 2.0
    text_lengths = np.maximum([len(t) for t in texts], 1.0)

    delta_x = np.abs(centers_x[:, None] - centers_x[None, :]) / max(width, 1.0)
    delta_y = np.abs(centers_y[:, None] - centers_y[None, :]) / max(height, 1.0)
    aspect_ratios = np.broadcast_to((widths / heights)[:, None], delta_x.shape)
    height_ratios = heights[None, :] / heights[:, None]
    width_ratios = widths[None, :] / heights[:, None]
    length_ratios = text_lengths[None, :] / text_lengths[:, None]

    return np.stack(
        [delta_x, delta_y, aspect_ratios, height_ratios, width_ratios, length_ratios],
        axis=-1,
    )


class PickDataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        data_dir: Path,
        char2id: dict[str, int],
        tag2id: dict[str, int],
        max_boxes: int = 256,
        max_text_len: int = 64,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.data_dir = data_dir
        self.char2id = char2id
        self.tag2id = tag2id
        self.max_boxes = max_boxes
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        with open(data_file, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, image_file: Path) -> torch.Tensor:
        image = Image.open(image_file).convert("RGB")
        width, height = self.image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)

        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return (image_tensor - self.image_mean) / self.image_std

    def _build_char_tags(self, label: str, text_len: int) -> list[int]:
        if label == "O" or text_len == 0:
            return [0] * text_len

        if label.startswith("B-"):
            entity = label[2:]
            b_tag_id = self.tag2id[label]
            i_tag_id = self.tag2id.get(f"I-{entity}", b_tag_id)
            return [b_tag_id] + [i_tag_id] * (text_len - 1)

        i_tag_id = self.tag2id[label]
        return [i_tag_id] * text_len

    def _process_ocr(
        self, ocr: list[dict]
    ) -> tuple[list[list[float]], list[str], list[list[int]], list[int], list[list[int]]]:
        width, height = self.image_size
        boxes: list[list[float]] = []
        texts: list[str] = []
        input_ids: list[list[int]] = []
        lengths: list[int] = []
        labels: list[list[int]] = []

        for item in ocr[:self.max_boxes]:
            x1, y1, x2, y2 = item["box"]

            x1 = max(0.0, min(float(width), x1 / 1000.0 * width))
            y1 = max(0.0, min(float(height), y1 / 1000.0 * height))
            x2 = max(0.0, min(float(width), x2 / 1000.0 * width))
            y2 = max(0.0, min(float(height), y2 / 1000.0 * height))

            if x2 <= x1:
                x2 = min(float(width), x1 + 1.0)
                x1 = min(x1, x2 - 1.0)

            if y2 <= y1:
                y2 = min(float(height), y1 + 1.0)
                y1 = min(y1, y2 - 1.0)

            boxes.append([x1, y1, x2, y2])

            text = item["text"]
            texts.append(text)

            text_len = min(len(text), self.max_text_len)
            pad_len = self.max_text_len - text_len
            lengths.append(text_len)

            char_ids = [self.char2id.get(c, 1) for c in text[:text_len]]
            input_ids.append(char_ids + [0] * pad_len)

            char_tags = self._build_char_tags(item["label"], text_len)
            labels.append(char_tags + [0] * pad_len)

        return boxes, texts, input_ids, lengths, labels

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._process_image(self.data_dir / sample["image_path"])
        boxes, texts, input_ids, lengths, labels = self._process_ocr(sample["ocr"])
        relations = compute_relations(boxes, texts, self.image_size[0], self.image_size[1])

        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "relations": torch.from_numpy(relations),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "num_boxes": len(boxes),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_boxes = max(item["num_boxes"] for item in batch)
    max_text_len = batch[0]["input_ids"].shape[1]

    images = torch.stack([item["image"] for item in batch], dim=0)
    boxes = torch.zeros(batch_size, max_boxes, 4, dtype=torch.float32)
    input_ids = torch.zeros(batch_size, max_boxes, max_text_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, max_boxes, max_text_len, dtype=torch.bool)
    box_mask = torch.zeros(batch_size, max_boxes, dtype=torch.bool)
    relations = torch.zeros(batch_size, max_boxes, max_boxes, 6, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_boxes, max_text_len, dtype=torch.long)

    for batch_idx, item in enumerate(batch):
        num_boxes = item["num_boxes"]
        boxes[batch_idx, :num_boxes] = item["boxes"]
        input_ids[batch_idx, :num_boxes] = item["input_ids"]
        box_mask[batch_idx, :num_boxes] = True
        relations[batch_idx, :num_boxes, :num_boxes] = item["relations"]
        labels[batch_idx, :num_boxes] = item["labels"]

        for box_idx, text_len in enumerate(item["lengths"]):
            if text_len > 0:
                attention_mask[batch_idx, box_idx, :text_len] = False

    return {
        "images": images,
        "boxes": boxes,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "box_mask": box_mask,
        "relations": relations,
        "labels": labels,
    }


def align_labels(
    pred_ids: list[list[int]],
    id2tag: dict[int, str],
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for box_idx, tag_ids in enumerate(pred_ids):
        tags = [id2tag[tag_id] for tag_id in tag_ids if tag_id in id2tag]
        tag = next((t for t in tags if t.startswith("B-")), None)
        if tag is None:
            tag = next((t for t in tags if t.startswith("I-")), "O")
        labels[box_idx] = tag
    return labels


def extract_entities(
    ocr: list[dict],
    labels: dict[int, str],
) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    current_entity = None
    current_texts: list[str] = []

    for box_idx, item in enumerate(ocr):
        label = labels.get(box_idx, "O")
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
    all_pred_ids: list[list[list[int]]],
    samples: list[dict],
    id2tag: dict[int, str],
    max_boxes: int = 256,
) -> list[dict[str, list[str]]]:
    preds: list[dict[str, list[str]]] = []
    for pred_ids, sample in zip(all_pred_ids, samples):
        ocr = sample["ocr"][:max_boxes]
        labels = align_labels(pred_ids, id2tag)
        preds.append(extract_entities(ocr, labels))

    return preds


def predict_sample(
    image: Image.Image,
    ocr: list[dict],
    model: torch.nn.Module,
    char2id: dict[str, int],
    id2tag: dict[int, str],
    device: torch.device,
    max_boxes: int = 256,
    max_text_len: int = 64,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, list[str]]:
    image = image.convert("RGB")
    width, height = image_size
    image = image.resize((width, height), Image.Resampling.BILINEAR)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    images = ((image_tensor - mean) / std).unsqueeze(0).to(device)

    boxes: list[list[float]] = []
    texts: list[str] = []
    input_ids: list[list[int]] = []
    lengths: list[int] = []

    for item in ocr[:max_boxes]:
        x1, y1, x2, y2 = item["box"]
        x1 = max(0.0, min(float(width), x1 / 1000.0 * width))
        y1 = max(0.0, min(float(height), y1 / 1000.0 * height))
        x2 = max(0.0, min(float(width), x2 / 1000.0 * width))
        y2 = max(0.0, min(float(height), y2 / 1000.0 * height))

        if x2 <= x1:
            x2 = min(float(width), x1 + 1.0)
            x1 = min(x1, x2 - 1.0)

        if y2 <= y1:
            y2 = min(float(height), y1 + 1.0)
            y1 = min(y1, y2 - 1.0)

        boxes.append([x1, y1, x2, y2])

        text = item["text"]
        texts.append(text)

        text_len = min(len(text), max_text_len)
        pad_len = max_text_len - text_len
        lengths.append(text_len)

        char_ids = [char2id.get(c, 1) for c in text[:text_len]]
        input_ids.append(char_ids + [0] * pad_len)

    num_boxes = len(boxes)
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0).to(device)
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    box_mask = torch.ones(1, num_boxes, dtype=torch.bool, device=device)
    attention_mask = torch.ones(1, num_boxes, max_text_len, dtype=torch.bool, device=device)
    for box_idx, text_len in enumerate(lengths):
        if text_len > 0:
            attention_mask[0, box_idx, :text_len] = False

    relations = compute_relations(boxes, texts, width, height)
    relations_tensor = torch.tensor(relations, dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        _, preds = model(
            images=images,
            boxes=boxes_tensor,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            box_mask=box_mask,
            relations=relations_tensor,
            return_preds=True,
        )

    labels = align_labels(preds[0], id2tag)
    return extract_entities(ocr[:max_boxes], labels)