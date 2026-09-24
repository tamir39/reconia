"""RoBERTa document dataset and tokenization pipeline for invoice KIE.

Converts intermediate Document objects into tokenized subwords aligned with:
1. 2D spatial bounding boxes normalized to [0, 1000] for each token.
2. BIO classification labels aligned to the first subword of each OCR word (with -100 for subsequent subwords and special tokens).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from app.data.schema import Document
from app.graph.labels import label_document


def extract_word_bboxes_and_labels(
    document: Document,
    label_vocabulary: dict[str, int],
    include_line_items: bool = True,
) -> tuple[list[str], list[list[int]], list[int]]:
    """Extract ordered words, [0, 1000] normalized boxes, and BIO label IDs from a Document."""
    words = document.words
    if not words:
        return ([], [], [])

    tags, _ = label_document(document, include_line_items=include_line_items)

    word_texts: list[str] = []
    word_boxes: list[list[int]] = []
    word_labels: list[int] = []

    for word, tag in zip(words, tags):
        text = word.text.strip()
        if not text:
            text = "[UNK]"

        # Clamp and normalize coordinates to [0, 1000]
        x0 = int(round(max(0.0, min(1.0, word.bbox.x0)) * 1000))
        y0 = int(round(max(0.0, min(1.0, word.bbox.y0)) * 1000))
        x1 = int(round(max(0.0, min(1.0, word.bbox.x1)) * 1000))
        y1 = int(round(max(0.0, min(1.0, word.bbox.y1)) * 1000))

        # Ensure non-degenerate box
        if x1 < x0:
            x1 = x0
        if y1 < y0:
            y1 = y0

        label_id = label_vocabulary.get(tag, 0)

        word_texts.append(text)
        word_boxes.append([x0, y0, x1, y1])
        word_labels.append(label_id)

    return word_texts, word_boxes, word_labels


def encode_document_for_roberta(
    document: Document,
    tokenizer: PreTrainedTokenizerFast,
    label_vocabulary: dict[str, int],
    max_seq_length: int = 512,
    include_line_items: bool = True,
) -> dict[str, torch.Tensor]:
    """Encode a single Document into PyTorch tensors for RoBERTaWithLayout."""
    word_texts, word_boxes, word_labels = extract_word_bboxes_and_labels(
        document, label_vocabulary, include_line_items=include_line_items
    )

    if not word_texts:
        # Fallback empty document
        return {
            "input_ids": torch.zeros((max_seq_length,), dtype=torch.long),
            "attention_mask": torch.zeros((max_seq_length,), dtype=torch.long),
            "bbox": torch.zeros((max_seq_length, 4), dtype=torch.long),
            "labels": torch.full((max_seq_length,), -100, dtype=torch.long),
        }

    encoding = tokenizer(
        word_texts,
        is_split_into_words=True,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)  # (seq_len,)
    attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)

    # Extract word_ids for subword alignment
    word_ids = encoding.word_ids(batch_index=0)

    bbox_list: list[list[int]] = []
    labels_list: list[int] = []
    previous_word_idx: int | None = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens: <s>, </s>, <pad>
            bbox_list.append([0, 0, 0, 0])
            labels_list.append(-100)
        else:
            bbox_list.append(word_boxes[word_idx])
            if word_idx != previous_word_idx:
                # First subword of the word: assign actual BIO label
                labels_list.append(word_labels[word_idx])
            else:
                # Subsequent subwords: mask out with -100
                labels_list.append(-100)
        previous_word_idx = word_idx

    bbox_tensor = torch.tensor(bbox_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bbox_tensor,
        "labels": labels_tensor,
    }


WINDOW_POLICY = "all-windows-first-subword-v1"


def encode_document_windows(document, tokenizer, label_vocabulary, max_seq_length=512, include_line_items=True):
    """Cover every OCR word, supervising its first subword exactly once."""
    texts, boxes, labels = extract_word_bboxes_and_labels(document, label_vocabulary, include_line_items)
    if not texts:
        return []
    encoded = tokenizer(texts, is_split_into_words=True, max_length=max_seq_length,
                        padding="max_length", truncation=True, return_overflowing_tokens=True,
                        stride=0, return_tensors="pt")
    windows, seen = [], set()
    for i in range(len(encoded["input_ids"])):
        word_ids = encoded.word_ids(batch_index=i)
        window_boxes, window_labels, word_indices = [], [], []
        for word_id in word_ids:
            window_boxes.append(boxes[word_id] if word_id is not None else [0, 0, 0, 0])
            first = word_id is not None and word_id not in seen
            window_labels.append(labels[word_id] if first else -100)
            word_indices.append(word_id if first else -1)
            if word_id is not None:
                seen.add(word_id)
        if all(idx == -1 for idx in word_indices):
            continue
        windows.append({"input_ids": encoded["input_ids"][i],
                        "attention_mask": encoded["attention_mask"][i],
                        "bbox": torch.tensor(window_boxes, dtype=torch.long),
                        "labels": torch.tensor(window_labels, dtype=torch.long),
                        "word_indices": torch.tensor(word_indices, dtype=torch.long)})
    if seen != set(range(len(texts))):
        raise ValueError("Tokenizer failed to cover every OCR word")
    return windows


class RoBERTaInvoiceDataset(Dataset):
    """PyTorch Dataset caching encoded documents for fast batch iteration."""

    def __init__(
        self,
        documents: list[Document],
        tokenizer: PreTrainedTokenizerFast,
        label_vocabulary: dict[str, int],
        max_seq_length: int = 512,
        include_line_items: bool = True,
    ) -> None:
        self.documents = documents
        self.tokenizer = tokenizer
        self.label_vocabulary = label_vocabulary
        self.max_seq_length = max_seq_length
        self.include_line_items = include_line_items

        self.encoded_items: list[dict[str, torch.Tensor]] = []
        for doc_index, doc in enumerate(documents):
            windows = encode_document_windows(
                doc,
                tokenizer=self.tokenizer,
                label_vocabulary=self.label_vocabulary,
                max_seq_length=self.max_seq_length,
                include_line_items=self.include_line_items,
            )
            for encoded in windows:
                encoded["document_index"] = torch.tensor(doc_index, dtype=torch.long)
                self.encoded_items.append(encoded)

    def __len__(self) -> int:
        return len(self.encoded_items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encoded_items[idx]
