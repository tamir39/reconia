import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


class QwenDataset(Dataset):
    def __init__(self, data_file, processor, data_dir, *, fields, training=True, max_new_tokens=2048, max_length=8192):
        self.samples = [json.loads(line) for line in Path(data_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.processor, self.data_dir = processor, Path(data_dir)
        self.training, self.max_new_tokens = training, max_new_tokens
        self.max_length = max_length
        prompt = ("Extract receipt information. Return only a JSON object mapping each present field "
                  "to a list of strings. Preserve repeated values. Field names: " + json.dumps(fields))
        self.prefix = processor.apply_chat_template([
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ], tokenize=False, add_generation_prompt=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.data_dir / sample["image_path"]) as image:
            encoded = self.processor(text=self.prefix, images=image.convert("RGB"),
                                     return_tensors="pt", truncation=False)
        ids = encoded["input_ids"][0].tolist()
        if len(ids) + self.max_new_tokens > self.max_length:
            raise ValueError(f"Qwen input exceeds context budget: {sample['id']}")
        feature = {"pixel_values": encoded["pixel_values"], "image_grid_thw": encoded["image_grid_thw"]}
        if self.training:
            target = json.dumps(sample["gt"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            answer = self.processor.tokenizer(target, add_special_tokens=False)["input_ids"]
            answer += [self.processor.tokenizer.eos_token_id]
            if len(answer) > self.max_new_tokens:
                raise ValueError(f"Qwen target too long: {sample['id']}")
            feature["labels"] = [-100] * len(ids) + answer
            ids += answer
        feature["input_ids"], feature["attention_mask"] = ids, [1] * len(ids)
        return feature


class QwenDataCollator:
    def __init__(self, processor, training=True):
        self.pad_token_id, self.training = processor.tokenizer.pad_token_id, training

    def __call__(self, features):
        width = max(len(item["input_ids"]) for item in features)
        batch = {}
        keys = [("input_ids", self.pad_token_id), ("attention_mask", 0)]
        if self.training:
            keys.append(("labels", -100))
        for key, fill in keys:
            rows = []
            for item in features:
                padding = [fill] * (width - len(item[key]))
                rows.append(item[key] + padding if self.training else padding + item[key])
            batch[key] = torch.tensor(rows, dtype=torch.long)
        batch["pixel_values"] = torch.cat([item["pixel_values"] for item in features])
        batch["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in features])
        return batch
