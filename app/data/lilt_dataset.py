"""Page-separated OCR and bounding-box windows for LiLT."""

import torch
from torch.utils.data import Dataset

from app.data.roberta_dataset import extract_word_bboxes_and_labels

WINDOW_POLICY = "page-layout-all-windows-first-subword-v1"


def encode_lilt_windows(document, tokenizer, vocabulary, max_length=512):
    """Supervise each OCR word once, keeping page coordinates in separate windows."""
    texts, boxes, labels = extract_word_bboxes_and_labels(document, vocabulary)
    windows = []
    pages = sorted({word.page for word in document.words})
    for page in pages:
        if not 0 <= page < document.page_count:
            raise ValueError(f"Invalid page {page} in {document.doc_id}")
        indices = [i for i, word in enumerate(document.words) if word.page == page]
        encoded = tokenizer(
            [texts[i] for i in indices], boxes=[boxes[i] for i in indices],
            max_length=max_length, padding="max_length", truncation=True,
            return_overflowing_tokens=True, stride=0, return_tensors="pt",
        )
        seen = set()
        for row in range(len(encoded["input_ids"])):
            word_indices, window_labels = [], []
            for word_id in encoded.word_ids(batch_index=row):
                first = word_id is not None and word_id not in seen
                global_id = indices[word_id] if first else -1
                word_indices.append(global_id)
                window_labels.append(labels[global_id] if first else -100)
                if word_id is not None:
                    seen.add(word_id)
            if all(i < 0 for i in word_indices):
                continue
            windows.append({
                "input_ids": encoded["input_ids"][row],
                "attention_mask": encoded["attention_mask"][row],
                "bbox": encoded["bbox"][row],
                "labels": torch.tensor(window_labels, dtype=torch.long),
                "word_indices": torch.tensor(word_indices, dtype=torch.long),
            })
        if seen != set(range(len(indices))):
            raise ValueError("Tokenizer did not cover all OCR words on the page")
    return windows


class LiLTInvoiceDataset(Dataset):
    def __init__(self, documents, tokenizer, vocabulary):
        self.documents = documents
        self.items = []
        for index, doc in enumerate(documents):
            for item in encode_lilt_windows(doc, tokenizer, vocabulary):
                item["document_index"] = torch.tensor(index, dtype=torch.long)
                self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
