import json
from pathlib import Path
import re

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

SYSTEM_PROMPT = (
    "You are a receipt information extraction assistant. "
    "Extract structured data from receipt images and respond only in valid JSON format."
)


def format_prompt(labels: list[str] | None = None) -> str:
    fields: list[str] = []
    if labels:
        seen: set[str] = set()
        for lbl in labels:
            if lbl.startswith("B-"):
                field = lbl[2:]
                if field not in seen:
                    seen.add(field)
                    fields.append(field)

    if not fields:
        return "Extract all key information from this receipt image."

    field_list = ", ".join(fields)
    return (
        f"Extract the following fields from this receipt image: {field_list}. "
        f"If a field appears multiple times, use a list. If a field is not found, omit it."
    )


class QwenVLDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        data_file: Path,
        processor: AutoProcessor,
        labels: list[str] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.processor = processor
        self.labels = labels

        with open(data_file, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = self.data_dir / sample["image_path"]
        image = (
            Image.open(image_path).convert("RGB")
            if image_path.exists()
            else Image.new("RGB", (224, 224), color=(255, 255, 255))
        )

        prompt_text = format_prompt(self.labels)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        prompt_chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        gt = sample.get("gt", {})
        target_json = json.dumps(gt, ensure_ascii=False)
        messages.append({"role": "assistant", "content": target_json})

        full_chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        full_inputs = self.processor(
            [image],
            [full_chat],
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]

        prompt_inputs = self.processor(
            [image],
            [prompt_chat],
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": prompt_len,
        }

        if "pixel_values" in full_inputs:
            item["pixel_values"] = full_inputs["pixel_values"]

        if "image_grid_thw" in full_inputs:
            item["image_grid_thw"] = full_inputs["image_grid_thw"]

        return item


class QwenVLDataCollator:
    def __init__(self, processor: AutoProcessor) -> None:
        self.tokenizer = processor.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"].tolist() + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"].tolist() + [0] * pad_len)
            labels.append(item["labels"].tolist() + [-100] * pad_len)

        collated = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": torch.tensor([item["prompt_len"] for item in batch], dtype=torch.long),
        }

        if "pixel_values" in batch[0]:
            collated["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            collated["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

        return collated


def extract_entities(text: str) -> dict[str, list[str]]:
    text = text.strip()

    block = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    content = block.group(1) if block else text

    decoder = json.JSONDecoder()

    for position, character in enumerate(content):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(content[position:])
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        entities = {}

        for key, value in parsed.items():
            values = value if isinstance(value, list) else [value]
            clean_values = [
                item.strip()
                for item in values
                if isinstance(item, str) and item.strip()
            ]
            if clean_values:
                entities[key] = clean_values

        return entities

    return {}


def decode_predictions(
    generated_ids: torch.Tensor,
    processor: AutoProcessor,
) -> list[dict[str, list[str]]]:
    decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [extract_entities(text) for text in decoded_texts]


def predict_sample(
    image: Image.Image,
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    labels: list[str] | None = None,
    max_new_tokens: int = 512,
) -> dict[str, list[str]]:
    prompt_text = format_prompt(labels)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if SYSTEM_PROMPT:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    prompt_chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_inputs = processor(
        [image],
        [prompt_chat],
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = prompt_inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return extract_entities(decoded)