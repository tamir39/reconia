import json
from pathlib import Path
import re

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor


class DonutDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        data_file: Path,
        processor: AutoProcessor,
        task_prompt: str,
        max_length: int = 512,
    ) -> None:
        self.data_dir = data_dir
        self.processor = processor
        self.task_prompt = task_prompt
        self.max_length = max_length

        with open(data_file, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[int]]:
        sample = self.samples[idx]
        image_path = self.data_dir / sample["image_path"]
        image = (
            Image.open(image_path).convert("RGB")
            if image_path.exists()
            else Image.new("RGB", (1280, 960))
        )

        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"][0]

        gt = sample.get("gt", {})
        target_json = json.dumps(gt, ensure_ascii=False)
        target_text = f"{self.task_prompt}{target_json}{self.processor.tokenizer.eos_token}"

        target_ids = self.processor.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )
        prompt_ids = self.processor.tokenizer.encode(self.task_prompt, add_special_tokens=False)
        labels = [-100] * len(prompt_ids) + target_ids[len(prompt_ids):]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class DonutDataCollator:
    def __init__(self, processor: AutoProcessor) -> None:
        tokenizer = processor.tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["labels"]) for item in batch)
        pixel_values: list[torch.Tensor] = []
        labels: list[list[int]] = []

        for item in batch:
            pad_len = max_len - len(item["labels"])
            pixel_values.append(item["pixel_values"])
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def extract_entities(text: str) -> dict[str, list[str]]:
    text = re.sub(r"<[^>]+>", "", text).strip()

    block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    content = block.group(1) if block else text

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
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


def decode_predictions(
    generated_ids: torch.Tensor,
    processor: AutoProcessor,
) -> list[dict[str, list[str]]]:
    decoded_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    return [extract_entities(text) for text in decoded_texts]


def predict_sample(
    image: Image.Image,
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    dataset_name: str,
    max_length: int = 512,
) -> dict[str, list[str]]:
    encoding = processor(image.convert("RGB"), return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    task_prompt = f"<s_{dataset_name}>"
    decoder_input_ids = processor.tokenizer.encode(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )

    decoded = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    return extract_entities(decoded)