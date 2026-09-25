"""CORD-v2 dataset adapter.

Converts the CORD-v2 receipt dataset into the intermediate Document representation.
Reads the dataset from disk (Hugging Face datasets format) and transforms
nested ground_truth lines/words and coordinates into normalized BBox, Word, and Field objects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from app.data.paths import REPO_ROOT, WORKSPACE_ROOT, resolve_dataset_path
from app.data.schema import BBox, Document, Field, Word


class CordAdapter:
    """Reads CORD-v2 into the intermediate representation."""

    name = "cord"

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        candidates = [
            WORKSPACE_ROOT / "cord",
            REPO_ROOT / "data" / "cord",
            Path("cord"),
        ]
        self.root = resolve_dataset_path(
            env_var_name="CORD_DIR",
            relative_candidates=candidates,
            dataset_name="CORD-v2",
            explicit_path=root,
        )

        from datasets import load_from_disk
        self._dataset = load_from_disk(str(self.root))

    def _normalize_split(self, split: str) -> str:
        if split == "val" and "validation" in self._dataset:
            return "validation"
        if split == "validation" and "val" in self._dataset:
            return "val"
        return split

    def splits(self) -> list[str]:
        return list(self._dataset.keys())

    def document_ids(self, split: str) -> list[str]:
        norm_split = self._normalize_split(split)
        if norm_split not in self._dataset:
            raise ValueError(f"unknown split {split!r}; available: {self.splits()}")
        count = len(self._dataset[norm_split])
        return [f"{norm_split}_{i}" for i in range(count)]

    def _parse_doc_id(self, doc_id: str) -> tuple[str, int]:
        """Parse doc_id format: e.g. 'train_42', 'val_5', 'validation_5', or integer index."""
        if "_" in doc_id:
            split, _, idx_str = doc_id.rpartition("_")
            norm_split = self._normalize_split(split)
            if norm_split in self._dataset:
                return norm_split, int(idx_str)
        # Fallback: assume train
        return "train", int(doc_id)

    def load(self, doc_id: str) -> Document:
        split, idx = self._parse_doc_id(doc_id)
        if split not in self._dataset:
            raise ValueError(f"unknown split {split!r}; available: {self.splits()}")
        sample = self._dataset[split][idx]

        gt_raw = sample.get("ground_truth", "{}")
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        meta = gt.get("meta", {})

        # Determine image dimensions for coordinate normalization
        image_obj = sample.get("image")
        if image_obj is not None and hasattr(image_obj, "size"):
            img_w, img_h = image_obj.size
        elif "image_size" in meta:
            img_w = float(meta["image_size"].get("width", 1))
            img_h = float(meta["image_size"].get("height", 1))
        else:
            img_w, img_h = 1000.0, 1000.0

        words: list[Word] = []
        fields: list[Field] = []

        for line in gt.get("valid_line", []):
            category = str(line.get("category", "")).strip()
            group_id = line.get("group_id")
            is_line_item = category.startswith(("menu.", "void_menu."))

            line_words: list[Word] = []
            for w in line.get("words", []):
                quad = w.get("quad", {})
                xs = [quad.get(k, 0) for k in ("x1", "x2", "x3", "x4")]
                ys = [quad.get(k, 0) for k in ("y1", "y2", "y3", "y4")]

                x0 = max(0.0, min(1.0, min(xs) / img_w))
                y0 = max(0.0, min(1.0, min(ys) / img_h))
                x1 = max(0.0, min(1.0, max(xs) / img_w))
                y1 = max(0.0, min(1.0, max(ys) / img_h))

                if x1 <= x0:
                    x1 = min(1.0, x0 + 0.001)
                if y1 <= y0:
                    y1 = min(1.0, y0 + 0.001)

                text = str(w.get("text", "")).strip()
                if text:
                    line_words.append(
                        Word(text=text, bbox=BBox(x0, y0, x1, y1), page=0, confidence=1.0)
                    )

            if line_words:
                words.extend(line_words)
                fx0 = min(w.bbox.x0 for w in line_words)
                fy0 = min(w.bbox.y0 for w in line_words)
                fx1 = max(w.bbox.x1 for w in line_words)
                fy1 = max(w.bbox.y1 for w in line_words)

                fields.append(
                    Field(
                        fieldtype=category,
                        bbox=BBox(fx0, fy0, fx1, fy1),
                        page=0,
                        text=" ".join(w.text for w in line_words),
                        line_item_id=group_id if is_line_item else None,
                    )
                )

        return Document(
            doc_id=doc_id,
            words=words,
            fields=fields,
            page_count=1,
            source=self.name,
        )

    def iter_documents(self, split: str, limit: int | None = None) -> Iterator[Document]:
        ids = self.document_ids(split)
        if limit is not None:
            ids = ids[:limit]
        for doc_id in ids:
            yield self.load(doc_id)

    def get_image(self, doc_id: str, page: int = 0):
        """Return PIL Image for the document from Hugging Face dataset."""
        if page != 0:
            raise ValueError("CORD receipts have exactly one page")
        split, idx = self._parse_doc_id(doc_id)
        sample = self._dataset[split][idx]
        img = sample.get("image")
        if img is not None:
            return img.convert("RGB")
        raise FileNotFoundError(f"Missing CORD image: {doc_id}")
