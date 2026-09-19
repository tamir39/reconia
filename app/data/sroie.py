"""SROIE dataset adapter.

Converts the ICDAR 2019 SROIE receipt dataset into the intermediate Document representation.
Splits the 626 documents into unbiased train (500), val (63), and test (63) sets using
a seeded shuffle (random seed 42) saved to sroie_splits.json for reproducibility.
Reads image dimensions, OCR word boxes, and matches key fields (company, date, address, total).
"""

from __future__ import annotations

import json
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator

from PIL import Image

from app.data.paths import REPO_ROOT, WORKSPACE_ROOT, resolve_dataset_path
from app.data.schema import BBox, Document, Field, Word

SPLITS_FILE = Path(__file__).resolve().parent / "sroie_splits.json"

RANDOM_SEED = 42
TRAIN_RATIO = 500
VAL_RATIO = 63


def _clean_text(text: str) -> str:
    """Normalize text for fuzzy matching by removing punctuation and whitespace."""
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


class SroieAdapter:
    """Reads ICDAR 2019 SROIE into the intermediate representation."""

    name = "sroie"

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        splits_path: str | os.PathLike[str] | None = None,
    ) -> None:
        candidates = [
            WORKSPACE_ROOT / "sroie" / "data",
            REPO_ROOT / "data" / "sroie" / "data",
            WORKSPACE_ROOT / "sroie",
            Path("sroie/data"),
        ]
        self.root = resolve_dataset_path(
            env_var_name="SROIE_DIR",
            relative_candidates=candidates,
            dataset_name="SROIE",
            explicit_path=root,
        )

        self.splits_file = Path(splits_path) if splits_path else SPLITS_FILE
        self._splits: dict[str, list[str]] = self._load_or_create_splits()

    def _load_or_create_splits(self) -> dict[str, list[str]]:
        """Load deterministic splits or create them using a seeded shuffle to prevent bias."""
        if self.splits_file.is_file():
            with self.splits_file.open(encoding="utf-8") as fh:
                return json.load(fh)

        # Collect all valid document IDs (must exist in box and key)
        box_dir = self.root / "box"
        key_dir = self.root / "key"
        if not box_dir.is_dir() or not key_dir.is_dir():
            return {"train": [], "val": [], "test": []}

        all_ids = sorted(
            p.stem for p in box_dir.glob("*.csv") if (key_dir / f"{p.stem}.json").is_file()
        )

        rng = random.Random(RANDOM_SEED)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)

        train_ids = sorted(shuffled[:TRAIN_RATIO])
        val_ids = sorted(shuffled[TRAIN_RATIO : TRAIN_RATIO + VAL_RATIO])
        test_ids = sorted(shuffled[TRAIN_RATIO + VAL_RATIO :])

        splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        # Save splits file for team reproducibility
        self.splits_file.parent.mkdir(parents=True, exist_ok=True)
        with self.splits_file.open("w", encoding="utf-8") as fh:
            json.dump(splits, fh, indent=2)

        return splits

    def splits(self) -> list[str]:
        return [s for s in ("train", "val", "test") if self._splits.get(s)]

    def document_ids(self, split: str) -> list[str]:
        if split not in self._splits:
            raise ValueError(f"unknown split {split!r}; available: {self.splits()}")
        return self._splits[split]

    def _get_image_size(self, doc_id: str) -> tuple[float, float]:
        """Read image dimensions from img/{doc_id}.jpg."""
        img_path = self.root / "img" / f"{doc_id}.jpg"
        if img_path.is_file():
            with Image.open(img_path) as img:
                return float(img.width), float(img.height)
        return 1000.0, 1000.0

    def load(self, doc_id: str) -> Document:
        box_path = self.root / "box" / f"{doc_id}.csv"
        key_path = self.root / "key" / f"{doc_id}.json"

        if not box_path.is_file():
            raise FileNotFoundError(f"Missing OCR file {box_path}")
        if not key_path.is_file():
            raise FileNotFoundError(f"Missing key annotation file {key_path}")

        img_w, img_h = self._get_image_size(doc_id)

        with box_path.open(encoding="utf-8", errors="ignore") as fh:
            raw_lines = [line.strip().split(",", 8) for line in fh if line.strip()]

        with key_path.open(encoding="utf-8", errors="ignore") as fh:
            key_data = json.load(fh)

        words: list[Word] = []
        line_word_groups: list[tuple[str, list[Word]]] = []

        for b in raw_lines:
            if len(b) < 9:
                continue
            coords = [float(x) for x in b[:8]]
            text = b[8].strip()
            if not text:
                continue

            xs = [coords[0], coords[2], coords[4], coords[6]]
            ys = [coords[1], coords[3], coords[5], coords[7]]

            x0 = max(0.0, min(1.0, min(xs) / img_w))
            y0 = max(0.0, min(1.0, min(ys) / img_h))
            x1 = max(0.0, min(1.0, max(xs) / img_w))
            y1 = max(0.0, min(1.0, max(ys) / img_h))

            if x1 <= x0:
                x1 = min(1.0, x0 + 0.001)
            if y1 <= y0:
                y1 = min(1.0, y0 + 0.001)

            tokens = text.split()
            if not tokens:
                continue

            tot_chars = sum(len(t) for t in tokens) + (len(tokens) - 1)
            cur_off = 0
            cur_line_words: list[Word] = []

            for t in tokens:
                t_len = len(t)
                wx0 = x0 + (x1 - x0) * (cur_off / tot_chars) if tot_chars > 0 else x0
                wx1 = (
                    x0 + (x1 - x0) * ((cur_off + t_len) / tot_chars)
                    if tot_chars > 0
                    else x1
                )
                if wx1 <= wx0:
                    wx1 = min(1.0, wx0 + 0.001)
                cur_off += t_len + 1

                w = Word(text=t, bbox=BBox(wx0, y0, wx1, y1), page=0, confidence=1.0)
                words.append(w)
                cur_line_words.append(w)

            line_word_groups.append((text, cur_line_words))

        # Match key fields (company, date, address, total)
        c_comp = _clean_text(str(key_data.get("company", "")))
        c_date = _clean_text(str(key_data.get("date", "")))
        c_addr = _clean_text(str(key_data.get("address", "")))
        c_tot = _clean_text(str(key_data.get("total", "")))

        fields: list[Field] = []

        # 1. Date (token level)
        if c_date:
            for line_text, lwords in line_word_groups:
                for w in lwords:
                    c_w = _clean_text(w.text)
                    if c_w and (
                        c_w == c_date
                        or (len(c_w) >= 4 and (c_w in c_date or c_date in c_w))
                    ):
                        fields.append(
                            Field(fieldtype="date", bbox=w.bbox, page=0, text=w.text)
                        )
                        break
                if any(f.fieldtype == "date" for f in fields):
                    break

        # 2. Total (token level, prioritize tokens on lines mentioning total/amount/due)
        if c_tot:
            tot_candidates: list[tuple[Word, bool]] = []
            for line_text, lwords in line_word_groups:
                is_tot_line = any(
                    k in line_text.lower()
                    for k in ("total", "amount", "due", "balance", "rm")
                )
                for w in lwords:
                    c_w = _clean_text(w.text)
                    if c_w and (c_w == c_tot or (len(c_w) >= 3 and c_w == c_tot)):
                        tot_candidates.append((w, is_tot_line))
            if tot_candidates:
                best_w = next(
                    (w for w, is_tl in tot_candidates if is_tl), tot_candidates[-1][0]
                )
                fields.append(
                    Field(fieldtype="total", bbox=best_w.bbox, page=0, text=best_w.text)
                )

        # 3. Company & Address (line level with distinct bounding boxes)
        for line_text, lwords in line_word_groups:
            c_line = _clean_text(line_text)
            if not c_line or len(c_line) < 4:
                continue

            score_comp = 0.0
            if c_comp:
                if c_line in c_comp:
                    score_comp = max(0.80, len(c_line) / len(c_comp))
                elif c_comp in c_line:
                    score_comp = max(0.80, len(c_comp) / len(c_line))
                else:
                    r = SequenceMatcher(None, c_line, c_comp).ratio()
                    if r >= 0.70:
                        score_comp = r

            score_addr = 0.0
            if c_addr:
                if c_line in c_addr:
                    score_addr = max(0.80, len(c_line) / len(c_addr))
                elif c_addr in c_line:
                    score_addr = max(0.80, len(c_addr) / len(c_line))
                else:
                    r = SequenceMatcher(None, c_line, c_addr).ratio()
                    if r >= 0.70:
                        score_addr = r

            best_type = None
            if score_comp > 0 and score_comp >= score_addr:
                best_type = "company"
            elif score_addr > 0 and score_addr > score_comp:
                best_type = "address"

            if best_type:
                fx0 = min(w.bbox.x0 for w in lwords)
                fy0 = min(w.bbox.y0 for w in lwords)
                fx1 = max(w.bbox.x1 for w in lwords)
                fy1 = max(w.bbox.y1 for w in lwords)
                fields.append(
                    Field(
                        fieldtype=best_type,
                        bbox=BBox(fx0, fy0, fx1, fy1),
                        page=0,
                        text=line_text,
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

    def image_path(self, doc_id: str) -> Path:
        """Location of the receipt scan image."""
        return self.root / "img" / f"{doc_id}.jpg"

    def get_image(self, doc_id: str, page: int = 0):
        """Return PIL Image for the receipt scan."""
        img_p = self.image_path(doc_id)
        if img_p.is_file():
            with Image.open(img_p) as img:
                return img.convert("RGB")
        return Image.new("RGB", (1000, 1000), color=(255, 255, 255))
