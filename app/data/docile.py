"""DocILE adapter.

Converts the DocILE `annotated-trainval` split into the intermediate representation.

Layout expected under DOCILE_DIR:
    annotations/<doc_id>.json    field_extractions, line_item_extractions
    ocr/<doc_id>.json            pages[].blocks[]...words with value + geometry
    pdfs/<doc_id>.pdf
    train.json, val.json         lists of document ids

OCR boxes come as [[x0, y0], [x1, y1]] in normalised coordinates. `snapped_geometry` is
preferred where present: DocILE snaps word boxes to the rendered text and uses the snapped
boxes in its own evaluation, so using them keeps our labels consistent with the benchmark.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from app.data.paths import REPO_ROOT, WORKSPACE_ROOT, resolve_dataset_path
from app.data.schema import BBox, Document, Field, Word

import random

SPLITS_FILE = Path(__file__).resolve().parent / "docile_splits.json"
RANDOM_SEED = 42
TRAIN_COUNT = 4680


def _bbox(pair: list[list[float]]) -> BBox:
    (x0, y0), (x1, y1) = pair
    return BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _bbox_from_flat(values: list[float]) -> BBox:
    x0, y0, x1, y1 = values
    return BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _iter_ocr_words(payload: dict) -> Iterator[tuple[int, str, BBox, float | None]]:
    """Walk nested OCR words in reading order."""
    for page in payload.get("pages", []):
        page_idx = page.get("page_idx", 0)
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    if "value" in word and "geometry" in word:
                        geometry = word.get("snapped_geometry") or word["geometry"]
                        text = str(word["value"]).strip()
                        if text:
                            yield page_idx, text, _bbox(geometry), word.get("confidence")


class DocileAdapter:
    """Reads DocILE documents into intermediate representation."""

    name = "docile"

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        test_root: str | os.PathLike[str] | None = None,
        splits_path: str | os.PathLike[str] | None = None,
    ) -> None:
        candidates = [
            WORKSPACE_ROOT / "docile" / "data" / "annotated-trainval",
            WORKSPACE_ROOT / "docile" / "data",
            REPO_ROOT / "data" / "docile" / "annotated-trainval",
            REPO_ROOT / "data" / "docile",
        ]
        self.root = resolve_dataset_path(
            env_var_name="DOCILE_DIR",
            relative_candidates=candidates,
            dataset_name="DocILE",
            explicit_path=root,
        )

        if test_root:
            self.test_root: Path | None = Path(test_root)
        else:
            test_candidates = [
                WORKSPACE_ROOT / "docile" / "data" / "test",
                self.root.parent / "test",
                self.root / "test",
            ]
            self.test_root = next((c for c in test_candidates if c.is_dir()), None)

        self.splits_file = Path(splits_path) if splits_path else SPLITS_FILE
        self._splits = self._load_or_create_splits()

    def _load_or_create_splits(self) -> dict[str, list[str]]:
        if self.splits_file.is_file():
            with self.splits_file.open(encoding="utf-8") as fh:
                return json.load(fh)

        train_p = self.root / "train.json"
        val_p = self.root / "val.json"
        if not train_p.is_file() or not val_p.is_file():
            return {}

        with train_p.open(encoding="utf-8") as fh:
            train_ids = json.load(fh)
        with val_p.open(encoding="utf-8") as fh:
            val_ids = json.load(fh)

        rng = random.Random(RANDOM_SEED)
        shuffled = list(train_ids)
        rng.shuffle(shuffled)

        splits = {
            "train": sorted(shuffled[:TRAIN_COUNT]),
            "val": val_ids,
            "test": sorted(shuffled[TRAIN_COUNT:]),
        }

        self.splits_file.parent.mkdir(parents=True, exist_ok=True)
        with self.splits_file.open("w", encoding="utf-8") as fh:
            json.dump(splits, fh, indent=2)

        return splits

    def splits(self) -> list[str]:
        base_splits = ["train", "val", "test"] if self._splits else ["train", "val"]
        if self.test_root and (self.test_root / "test.json").is_file():
            base_splits.append("test_official")
        return base_splits

    def has_ground_truth(self, split: str) -> bool:
        return split in ("train", "val", "test", "trainval")

    def document_ids(self, split: str) -> list[str]:
        if split in self._splits:
            return self._splits[split]
        if split == "test_official" and self.test_root:
            test_p = self.test_root / "test.json"
            if test_p.is_file():
                with test_p.open(encoding="utf-8") as fh:
                    return json.load(fh)
        raw_p = self.root / f"{split}.json"
        if raw_p.is_file():
            with raw_p.open(encoding="utf-8") as fh:
                return json.load(fh)
        raise ValueError(f"unknown split {split!r}; available: {self.splits()}")

    def _find_file(self, subdir: str, filename: str) -> Path:
        """Locate an annotation, ocr, or pdf file across root and test_root."""
        primary = self.root / subdir / filename
        if primary.is_file():
            return primary
        if self.test_root:
            secondary = self.test_root / subdir / filename
            if secondary.is_file():
                return secondary
        raise FileNotFoundError(f"Could not find {filename} in {self.root / subdir}")

    def load(self, doc_id: str) -> Document:
        ann_path = self._find_file("annotations", f"{doc_id}.json")
        ocr_path = self._find_file("ocr", f"{doc_id}.json")

        with ann_path.open(encoding="utf-8") as fh:
            annotation = json.load(fh)
        with ocr_path.open(encoding="utf-8") as fh:
            ocr = json.load(fh)

        words = [
            Word(text=text, bbox=box, page=page, confidence=conf)
            for page, text, box, conf in _iter_ocr_words(ocr)
        ]

        fields: list[Field] = []
        for entry in annotation.get("field_extractions", []):
            fields.append(
                Field(
                    fieldtype=entry["fieldtype"],
                    bbox=_bbox_from_flat(entry["bbox"]),
                    page=entry.get("page", 0),
                    text=entry.get("text"),
                )
            )
        for entry in annotation.get("line_item_extractions", []):
            fields.append(
                Field(
                    fieldtype=entry["fieldtype"],
                    bbox=_bbox_from_flat(entry["bbox"]),
                    page=entry.get("page", 0),
                    text=entry.get("text"),
                    line_item_id=entry.get("line_item_id"),
                )
            )

        return Document(
            doc_id=doc_id,
            words=words,
            fields=fields,
            page_count=len(ocr.get("pages", [])) or 1,
            source=self.name,
        )

    def iter_documents(self, split: str, limit: int | None = None) -> Iterator[Document]:
        ids = self.document_ids(split)
        if limit is not None:
            ids = ids[:limit]
        for doc_id in ids:
            yield self.load(doc_id)

    def pdf_path(self, doc_id: str) -> Path:
        """Location of the source PDF, needed for visual features later."""
        return self._find_file("pdfs", f"{doc_id}.pdf")

    def get_image(self, doc_id: str, page: int = 0):
        """Extract or render the page image from the source PDF."""
        import pypdf
        from PIL import Image

        pdf_p = self.pdf_path(doc_id)
        reader = pypdf.PdfReader(str(pdf_p))
        if page < len(reader.pages):
            p = reader.pages[page]
            if p.images:
                return p.images[0].image.convert("RGB")
        return Image.new("RGB", (1000, 1000), color=(255, 255, 255))
