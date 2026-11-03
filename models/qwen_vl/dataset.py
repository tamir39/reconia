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


def parse_json_entities(text: str) -> dict[str, list[str]]:
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    candidate_text = json_match.group(1) if json_match else text

    brace_match = re.search(r"\{.*\}", candidate_text, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
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


class QwenVLDataset(Dataset):
    def __init__(
        self,
        data_file: Path,
        processor: AutoProcessor,
        labels: list[str] | None = None,
        data_dir: Path | None = None,
        max_length: int = 2048,
    ) -> None:
        self.processor = processor
        self.labels = labels
        self.data_dir = data_dir
        self.max_length = max_length

        with open(data_file, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = None
        if self.data_dir and "image_path" in sample:
            image_path = self.data_dir / sample["image_path"]
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")

        if image is None:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        prompt_text = format_prompt(self.labels)
        gt_json = json.dumps(sample.get("gt", {}), ensure_ascii=False)

        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        })

        prompt_chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        full_messages = list(messages)
        full_messages.append({"role": "assistant", "content": gt_json})
        full_chat = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[full_chat],
            images=[image],
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        prompt_inputs = self.processor(
            text=[prompt_chat],
            images=[image],
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
        if "pixel_values" in inputs:
            item["pixel_values"] = inputs["pixel_values"][0]
        if "image_grid_thw" in inputs:
            item["image_grid_thw"] = inputs["image_grid_thw"][0]

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
            collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            collated["image_grid_thw"] = torch.stack([item["image_grid_thw"] for item in batch], dim=0)

        return collated


def decode_predictions(
    generated_ids: torch.Tensor,
    processor: AutoProcessor,
) -> list[dict[str, list[str]]]:
    decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [parse_json_entities(text) for text in decoded_texts]


def predict_sample(
    image: Image.Image,
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    labels: list[str] | None = None,
    max_new_tokens: int = 512,
) -> dict[str, list[str]]:
    prompt_text = format_prompt(labels)
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    })

    prompt_chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_chat], images=[image], return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return parse_json_entities(decoded)