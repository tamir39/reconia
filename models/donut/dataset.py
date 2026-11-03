import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def parse_json_entities(text: str) -> dict[str, list[str]]:
    text = re.sub(r"<[^>]+>", "", text).strip()

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                entities: dict[str, list[str]] = {}
                for key, val in parsed.items():
                    if isinstance(val, list):
                        entities[str(key)] = [str(v).strip() for v in val if str(v).strip()]
                    elif isinstance(val, (str, int, float)):
                        val_str = str(val).strip()
                        if val_str:
                            entities[str(key)] = [val_str]
                return entities
        except Exception:
            pass

    entities: dict[str, list[str]] = {}
    pattern = r'"([^"]+)":\s*(?:\[([^\]]*)\]|"([^"]*)")'
    for match in re.finditer(pattern, text):
        key = match.group(1).strip()
        if match.group(2) is not None:
            raw_items = re.findall(r'"([^"]*)"', match.group(2))
            clean_items = [item.strip() for item in raw_items if item.strip()]
            if clean_items:
                entities.setdefault(key, []).extend(clean_items)
        elif match.group(3) is not None:
            val = match.group(3).strip()
            if val:
                entities.setdefault(key, []).append(val)

    return entities


def process_donut_image(
    image: Image.Image,
    target_size: tuple[int, int] = (1280, 960),
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    target_w, target_h = target_size
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    img_resized = img_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(img_resized, (0, 0))

    tensor = torch.from_numpy(np.array(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return (tensor - mean) / std


class DonutDataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        data_dir: Path | None = None,
        max_length: int = 512,
        image_size: tuple[int, int] = (1280, 960),
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.max_length = max_length
        self.image_size = image_size
        self.task_prompt = f"<s_{dataset_name}>"

        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        with open(data_file, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, sample: dict) -> torch.Tensor:
        if self.data_dir and "image_path" in sample:
            image_path = self.data_dir / sample["image_path"]
            if image_path.exists():
                with Image.open(image_path) as img:
                    return process_donut_image(
                        img,
                        self.image_size,
                        self.image_mean,
                        self.image_std,
                    )

        return torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[int]]:
        sample = self.samples[idx]
        gt = sample.get("gt", {})
        target_json = json.dumps(gt, ensure_ascii=False)
        target_text = f"{self.task_prompt}{target_json}</s>"

        prompt_ids = self.tokenizer.encode(self.task_prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False, max_length=self.max_length, truncation=True)

        labels = [-100] * len(prompt_ids) + target_ids[len(prompt_ids):]

        pixel_values = self._process_image(sample)

        return {
            "pixel_values": pixel_values,
            "input_ids": target_ids,
            "labels": labels,
        }


class DonutDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        labels: list[list[int]] = []
        pixel_values: list[torch.Tensor] = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            labels.append(item["labels"] + [-100] * pad_len)
            pixel_values.append(item["pixel_values"])

        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def decode_predictions(
    generated_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, list[str]]]:
    decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    return [parse_json_entities(text) for text in decoded_texts]


def predict_sample(
    image: Image.Image,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    dataset_name: str,
    image_size: tuple[int, int] = (1280, 960),
    max_length: int = 512,
) -> dict[str, list[str]]:
    pixel_values = process_donut_image(image, image_size).unsqueeze(0).to(device)

    task_prompt = f"<s_{dataset_name}>"
    decoder_input_ids = tokenizer.encode(task_prompt, add_special_tokens=False, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return parse_json_entities(decoded)