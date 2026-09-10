"""Intermediate representation shared by every dataset.

Each dataset adapter converts its own format into these types. Everything downstream -
graph construction, models, evaluation - reads only this representation, so adding a
dataset never requires touching the model or evaluation code.

Coordinates are always normalised to [0, 1] relative to the page, so documents of
different sizes and resolutions are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in normalised page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"degenerate box: {self}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def intersection_area(self, other: BBox) -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def covered_by(self, other: BBox) -> float:
        """Fraction of this box lying inside `other`.

        Used instead of IoU when deciding whether a word belongs to an annotated field:
        a field box may span several words, so IoU against any single word is small even
        when the word is entirely inside the field.
        """
        return self.intersection_area(other) / self.area if self.area > 0 else 0.0

    def vertical_overlap(self, other: BBox) -> float:
        """Fraction of the shorter box's height shared with the other box."""
        lo, hi = max(self.y0, other.y0), min(self.y1, other.y1)
        if hi <= lo:
            return 0.0
        shorter = min(self.height, other.height)
        return (hi - lo) / shorter if shorter > 0 else 0.0

    def horizontal_overlap(self, other: BBox) -> float:
        lo, hi = max(self.x0, other.x0), min(self.x1, other.x1)
        if hi <= lo:
            return 0.0
        narrower = min(self.width, other.width)
        return (hi - lo) / narrower if narrower > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Word:
    """One token produced by OCR. Becomes a node in the document graph."""

    text: str
    bbox: BBox
    page: int
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Field:
    """One annotated field. Supplies the labels.

    `line_item_id` is set only for fields belonging to a line item, and identifies which
    line item. It is the supervision signal for the LIR task.
    """

    fieldtype: str
    bbox: BBox
    page: int
    text: str | None = None
    line_item_id: int | None = None

    @property
    def is_line_item(self) -> bool:
        return self.line_item_id is not None


@dataclass(slots=True)
class Document:
    """A single document in the intermediate representation."""

    doc_id: str
    words: list[Word] = field(default_factory=list)
    fields: list[Field] = field(default_factory=list)
    page_count: int = 1
    source: str = ""

    def words_on(self, page: int) -> list[Word]:
        return [w for w in self.words if w.page == page]

    def fields_on(self, page: int) -> list[Field]:
        return [f for f in self.fields if f.page == page]

    @property
    def fieldtypes(self) -> set[str]:
        return {f.fieldtype for f in self.fields}


class DatasetAdapter(Protocol):
    """Contract every dataset must satisfy.

    Adding a dataset means implementing this and nothing else.
    """

    name: str

    def splits(self) -> list[str]:
        """Available split names, e.g. ["train", "val"]."""
        ...

    def document_ids(self, split: str) -> list[str]:
        ...

    def load(self, doc_id: str) -> Document:
        ...

    def iter_documents(self, split: str, limit: int | None = None) -> Iterator[Document]:
        ...
