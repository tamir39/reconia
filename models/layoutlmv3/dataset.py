import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from utils.token_data import TokenCollator, extract_entities


def encode_document(sample, tokenizer, max_length=512):
    return tokenizer(
        [item["text"] for item in sample["ocr"]],
        boxes=[item["box"] for item in sample["ocr"]],
        max_length=max_length, truncation=True,
    )


class LayoutLMv3Dataset(Dataset):
    def __init__(self, data_file, processor, label2id, data_dir, max_length=512):
        self.samples = [json.loads(line) for line in Path(data_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.processor, self.label2id = processor, label2id
        self.data_dir, self.max_length = Path(data_dir), max_length
        processor.image_processor.apply_ocr = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        encoding = encode_document(sample, self.processor.tokenizer, self.max_length)
        labels, previous = [], None
        for word in encoding.word_ids():
            labels.append(self.label2id[sample["ocr"][word]["label"]]
                          if word is not None and word != previous else -100)
            previous = word
        with Image.open(self.data_dir / sample["image_path"]) as image:
            pixels = self.processor.image_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"],
                "bbox": encoding["bbox"], "labels": labels, "pixel_values": pixels}


class LayoutLMv3DataCollator(TokenCollator):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, layout="layoutlmv3")


def decode_predictions(all_pred_ids, samples, tokenizer, id2label, max_length=512):
    if len(all_pred_ids) != len(samples):
        raise ValueError("Expected one LayoutLMv3 prediction per document")
    predictions = []
    for pred_ids, sample in zip(all_pred_ids, samples):
        encoding = encode_document(sample, tokenizer, max_length)
        if len(pred_ids) < len(encoding["input_ids"]):
            raise ValueError("Missing LayoutLMv3 token predictions")
        labels, previous = {}, None
        for token, word in zip(pred_ids, encoding.word_ids()):
            if word is not None and word != previous:
                labels[word] = id2label[token]
            previous = word
        # Truncated words remain O; evaluation still uses the complete document GT.
        predictions.append(extract_entities(sample["ocr"], labels))
    return predictions
