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
        max_length: int = 1024,
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
        image_file = self.data_dir / sample["image_path"]
        with Image.open(image_file) as img:
            image = img.convert("RGB")

        encoding = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            size={"height": 1280, "width": 960}
        )
        pixel_values = encoding["pixel_values"][0]

        gt = sample.get("gt", {})
        target_json = json.dumps(gt, ensure_ascii=False)
        target_text = f"{self.task_prompt}{target_json}{self.processor.tokenizer.eos_token}"

        target_ids = self.processor.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )
        prompt_ids = self.processor.tokenizer.encode(
            self.task_prompt,
            add_special_tokens=False
        )

        labels = target_ids.copy()
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": target_ids,
            "labels": labels,
        }


class DonutDataCollator:
    def __init__(self, processor: AutoProcessor) -> None:
        tokenizer = processor.tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        pixel_values: list[torch.Tensor] = []
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            pixel_values.append(item["pixel_values"])
            input_ids.append(
                item["input_ids"] + [self.pad_token_id] * pad_len
            )
            labels.append(
                item["labels"] + [-100] * pad_len
            )

        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def extract_entities(text: str) -> dict[str, list[str]]:
    text = re.sub(r"<[^>]+>", "", text).strip()

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
                str(item).strip()
                for item in values
                if item is not None and str(item).strip()
            ]
            if clean_values:
                entities[str(key)] = clean_values

        return entities

    return {}


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
    max_length: int = 1024,
) -> dict[str, list[str]]:
    encoding = processor(
        image,
        return_tensors="pt",
        do_resize=True,
        size={"height": 1280, "width": 960}
    )
    pixel_values = encoding["pixel_values"].to(device)

    task_prompt = f"<s_{dataset_name}>"
    prompt_ids = processor.tokenizer.encode(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    batch_size = pixel_values.shape[0]
    decoder_input_ids = prompt_ids.repeat(batch_size, 1)

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=1,
            use_cache=True,
        )

    decoded = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    return extract_entities(decoded)