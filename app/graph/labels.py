"""BIO labelling of OCR words from field annotations.

The survey of DocILE showed that 26% of annotated fields span more than one word, and
8.2% span six or more. Labelling each word independently therefore loses field boundaries:
two adjacent `vendor_name` words could belong to one field or to two different ones, and a
plain per-word label cannot tell them apart.

BIO tagging solves this. The first word of a field is tagged `B-<fieldtype>` and the rest
`I-<fieldtype>`, so consecutive fields of the same type remain separable when predictions
are decoded back into fields.

The survey also showed that 4.36% of fields match no OCR word at all. Those fields are
unreachable for any word-level model; `LabelStats.unmatched_fields` records them so the
ceiling is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from app.data.schema import Document, Field, Word

OUTSIDE = "O"

# A word counts as part of a field when at least this fraction of the word lies inside
# the field box. Half is deliberately permissive: OCR boxes and annotation boxes are
# drawn independently and rarely align exactly.
COVERAGE_THRESHOLD = 0.5


@dataclass
class LabelStats:
    """Diagnostics from a labelling pass. Reported alongside model results."""

    total_fields: int = 0
    matched_fields: int = 0
    unmatched_fields: int = 0
    total_words: int = 0
    labelled_words: int = 0
    multiword_fields: int = 0
    unmatched_by_type: dict[str, int] = dc_field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of fields reachable by word-level labelling: the model's ceiling."""
        return self.matched_fields / self.total_fields if self.total_fields else 0.0

    def merge(self, other: LabelStats) -> None:
        self.total_fields += other.total_fields
        self.matched_fields += other.matched_fields
        self.unmatched_fields += other.unmatched_fields
        self.total_words += other.total_words
        self.labelled_words += other.labelled_words
        self.multiword_fields += other.multiword_fields
        for k, v in other.unmatched_by_type.items():
            self.unmatched_by_type[k] = self.unmatched_by_type.get(k, 0) + v


def reading_order(words: list[Word], line_tolerance: float = 0.5) -> list[int]:
    """Indices of `words` in reading order: top to bottom, then left to right.

    Words are grouped into lines by vertical overlap rather than by exact y, because OCR
    boxes on the same visual line rarely share a y coordinate.
    """
    if not words:
        return []

    order = sorted(range(len(words)), key=lambda i: (words[i].bbox.y0, words[i].bbox.x0))
    lines: list[list[int]] = []
    for idx in order:
        placed = False
        for line in lines:
            if words[idx].bbox.vertical_overlap(words[line[0]].bbox) >= line_tolerance:
                line.append(idx)
                placed = True
                break
        if not placed:
            lines.append([idx])

    result: list[int] = []
    for line in lines:
        result.extend(sorted(line, key=lambda i: words[i].bbox.x0))
    return result


def _words_in_field(words: list[Word], target: Field) -> list[int]:
    return [
        i
        for i, w in enumerate(words)
        if w.page == target.page and w.bbox.covered_by(target.bbox) >= COVERAGE_THRESHOLD
    ]


def label_document(
    document: Document, *, include_line_items: bool = True
) -> tuple[list[str], LabelStats]:
    """Assign one BIO tag per word.

    Returns tags aligned with `document.words`, plus diagnostics.

    When two fields claim the same word, the smaller field wins: nested annotations occur
    (a line item total inside a table region), and the tighter box is the more specific
    description of that word.
    """
    words = document.words
    tags = [OUTSIDE] * len(words)
    claim_area = [float("inf")] * len(words)
    stats = LabelStats(total_words=len(words))

    targets = [
        f for f in document.fields if include_line_items or not f.is_line_item
    ]
    stats.total_fields = len(targets)

    for target in targets:
        members = _words_in_field(words, target)
        if not members:
            stats.unmatched_fields += 1
            stats.unmatched_by_type[target.fieldtype] = (
                stats.unmatched_by_type.get(target.fieldtype, 0) + 1
            )
            continue

        stats.matched_fields += 1
        if len(members) > 1:
            stats.multiword_fields += 1

        local = [words[i] for i in members]
        ordered = [members[j] for j in reading_order(local)]

        for position, word_idx in enumerate(ordered):
            if target.bbox.area >= claim_area[word_idx]:
                continue  # an equally or more specific field already claimed this word
            prefix = "B" if position == 0 else "I"
            tags[word_idx] = f"{prefix}-{target.fieldtype}"
            claim_area[word_idx] = target.bbox.area

    stats.labelled_words = sum(1 for t in tags if t != OUTSIDE)
    return tags, stats


def build_label_vocabulary(fieldtypes: list[str]) -> dict[str, int]:
    """Map BIO tags to contiguous ids, with `O` fixed at 0."""
    vocabulary = {OUTSIDE: 0}
    for name in sorted(fieldtypes):
        vocabulary[f"B-{name}"] = len(vocabulary)
        vocabulary[f"I-{name}"] = len(vocabulary)
    return vocabulary


def decode_fields(words: list[Word], tags: list[str]) -> list[Field]:
    """Reconstruct fields from per-word BIO tags.

    Inverse of `label_document`. A run starts at `B-` and continues through `I-` tags of
    the same type; an `I-` tag with no preceding `B-` is treated as a field start, since
    models emit that pattern and dropping it would silently lose predictions.
    """
    if len(words) != len(tags):
        raise ValueError("words and tags must have the same length")

    fields: list[Field] = []
    order = reading_order(words)
    current_type: str | None = None
    members: list[int] = []

    def flush() -> None:
        nonlocal current_type, members
        if current_type is None or not members:
            current_type, members = None, []
            return
        boxes = [words[i].bbox for i in members]
        fields.append(
            Field(
                fieldtype=current_type,
                bbox=type(boxes[0])(
                    min(b.x0 for b in boxes),
                    min(b.y0 for b in boxes),
                    max(b.x1 for b in boxes),
                    max(b.y1 for b in boxes),
                ),
                page=words[members[0]].page,
                text=" ".join(words[i].text for i in members),
            )
        )
        current_type, members = None, []

    for idx in order:
        tag = tags[idx]
        if tag == OUTSIDE:
            flush()
            continue
        prefix, _, name = tag.partition("-")
        if prefix == "B" or name != current_type:
            flush()
            current_type = name
            members = [idx]
        else:
            members.append(idx)
    flush()
    return fields
