import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


class DonutDataset(Dataset):
    def __init__(self, data_file, processor, data_dir, *, training=True, max_new_tokens=1535):
        self.samples = [json.loads(line) for line in Path(data_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.processor, self.data_dir = processor, Path(data_dir)
        self.training, self.max_new_tokens = training, max_new_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.data_dir / sample["image_path"]) as image:
            pixels = self.processor.image_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        tokenizer = self.processor.tokenizer
        feature = {"pixel_values": pixels, "decoder_input_ids": [tokenizer.bos_token_id]}
        if self.training:
            target = json.dumps(sample["gt"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            if len(ids) > self.max_new_tokens:
                raise ValueError(f"Donut target too long: {sample['id']} ({len(ids)} tokens)")
            feature["labels"] = ids
            feature["decoder_input_ids"] += ids[:-1]
        return feature


class DonutDataCollator:
    def __init__(self, processor, training=True):
        self.pad_token_id, self.training = processor.tokenizer.pad_token_id, training

    def __call__(self, features):
        width = max(len(item["decoder_input_ids"]) for item in features)
        batch = {"pixel_values": torch.stack([item["pixel_values"] for item in features])}
        batch["decoder_input_ids"] = torch.tensor([
            item["decoder_input_ids"] + [self.pad_token_id] * (width - len(item["decoder_input_ids"])) for item in features])
        batch["decoder_attention_mask"] = torch.tensor([
            [1] * len(item["decoder_input_ids"]) + [0] * (width - len(item["decoder_input_ids"])) for item in features])
        if self.training:
            batch["labels"] = torch.tensor([item["labels"] + [-100] * (width - len(item["labels"])) for item in features])
        return batch
