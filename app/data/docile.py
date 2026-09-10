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

from app.data.schema import BBox, Document, Field, Word

DEFAULT_DIR = "D:/Data/docile/data/annotated-trainval"


def _bbox(pair: list[list[float]]) -> BBox:
    (x0, y0), (x1, y1) = pair
    return BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _bbox_from_flat(values: list[float]) -> BBox:
    x0, y0, x1, y1 = values
    return BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _iter_ocr_words(payload: dict) -> Iterator[tuple[int, str, BBox, float | None]]:
    """Walk the nested OCR structure and yield every word.

    The nesting depth varies (pages -> blocks -> lines -> words), so the structure is
    traversed generically rather than by fixed key path.
    """
    for page in payload.get("pages", []):
        page_idx = page.get("page_idx", 0)
        stack: list = [page.get("blocks", [])]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "value" in node and "geometry" in node:
                    geometry = node.get("snapped_geometry") or node["geometry"]
                    text = str(node["value"])
                    if text.strip():
                        yield page_idx, text, _bbox(geometry), node.get("confidence")
                else:
                    stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend(node)


class DocileAdapter:
    """Reads DocILE into the intermediate representation."""

    name = "docile"

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root or os.environ.get("DOCILE_DIR", DEFAULT_DIR))
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"DocILE directory not found: {self.root}. "
                "Set DOCILE_DIR to the annotated-trainval directory."
            )

    def splits(self) -> list[str]:
        return [s for s in ("train", "val") if (self.root / f"{s}.json").is_file()]

    def document_ids(self, split: str) -> list[str]:
        path = self.root / f"{split}.json"
        if not path.is_file():
            raise ValueError(f"unknown split {split!r}; available: {self.splits()}")
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, doc_id: str) -> Document:
        with (self.root / "annotations" / f"{doc_id}.json").open(encoding="utf-8") as fh:
            annotation = json.load(fh)
        with (self.root / "ocr" / f"{doc_id}.json").open(encoding="utf-8") as fh:
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
        return self.root / "pdfs" / f"{doc_id}.pdf"
