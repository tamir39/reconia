"""Assemble a document into a graph ready for a node-classification model.

Nodes are OCR words. Node features are geometric and lexical only; the text itself is kept
alongside so a text encoder can be attached later without rebuilding the graph. Visual
features from a CNN are added in a later stage and simply extend the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.data.schema import Document, Word
from app.graph.edges import build_edges
from app.graph.labels import LabelStats, label_document

# Order is fixed and documented because the trained model depends on it.
FEATURE_NAMES = (
    "x0", "y0", "x1", "y1",
    "width", "height", "center_x", "center_y",
    "aspect_ratio", "area",
    "char_count", "is_numeric", "has_digit", "has_alpha", "has_punct", "is_upper",
    "ocr_confidence",
)
FEATURE_DIM = len(FEATURE_NAMES)

_PUNCT = set(".,;:/\\-()[]%$#*&@!?'\"")


@dataclass(slots=True)
class DocumentGraph:
    """A document as a graph. Backed by numpy so it stays framework independent."""

    doc_id: str
    texts: list[str]
    features: np.ndarray          # (num_nodes, FEATURE_DIM) float32
    edge_index: np.ndarray        # (2, num_edges) int64, undirected pairs stored once
    edge_type: np.ndarray         # (num_edges,) int64
    labels: np.ndarray            # (num_nodes,) int64, ids into the label vocabulary
    tags: list[str]               # human-readable BIO tags, aligned with labels
    pages: np.ndarray             # (num_nodes,) int64
    stats: LabelStats

    @property
    def num_nodes(self) -> int:
        return len(self.texts)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_undirected_index(self) -> np.ndarray:
        """Edges duplicated in both directions, the layout most GNN libraries expect."""
        if self.num_edges == 0:
            return np.zeros((2, 0), dtype=np.int64)
        return np.concatenate([self.edge_index, self.edge_index[::-1]], axis=1)

    def to_undirected(self) -> tuple[np.ndarray, np.ndarray]:
        """Edges and edge types duplicated symmetrically in both directions."""
        if self.num_edges == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        u_edge_index = np.concatenate([self.edge_index, self.edge_index[::-1]], axis=1)
        u_edge_type = np.concatenate([self.edge_type, self.edge_type], axis=0)
        return u_edge_index, u_edge_type


def word_features(word: Word) -> list[float]:
    box, text = word.bbox, word.text
    char_count = len(text)
    return [
        box.x0, box.y0, box.x1, box.y1,
        box.width, box.height, box.center[0], box.center[1],
        box.width / box.height if box.height > 0 else 0.0,
        box.area,
        # Length is scaled rather than raw so every feature stays in a comparable range.
        min(char_count, 40) / 40.0,
        float(text.replace(".", "").replace(",", "").isdigit()),
        float(any(c.isdigit() for c in text)),
        float(any(c.isalpha() for c in text)),
        float(any(c in _PUNCT for c in text)),
        float(text.isupper() and any(c.isalpha() for c in text)),
        float(word.confidence) if word.confidence is not None else 1.0,
    ]


def build_graph(
    document: Document,
    label_vocabulary: dict[str, int],
    *,
    include_line_items: bool = True,
    use_knn: bool = True,
    use_same_line: bool = True,
    use_vertical: bool = True,
    k: int = 6,
) -> DocumentGraph:
    """Convert one document into a graph.

    Unknown tags map to `O` rather than raising: a vocabulary built from the training
    split will not contain field types that appear only in validation, and losing those
    few labels is preferable to failing the run.
    """
    words = document.words
    tags, stats = label_document(document, include_line_items=include_line_items)

    if words:
        features = np.asarray([word_features(w) for w in words], dtype=np.float32)
        pages = np.asarray([w.page for w in words], dtype=np.int64)
    else:
        features = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        pages = np.zeros((0,), dtype=np.int64)

    pairs, types = build_edges(
        words,
        use_knn=use_knn,
        use_same_line=use_same_line,
        use_vertical=use_vertical,
        k=k,
    )
    if pairs:
        edge_index = np.asarray(pairs, dtype=np.int64).T
        edge_type = np.asarray(types, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    labels = np.asarray(
        [label_vocabulary.get(tag, 0) for tag in tags], dtype=np.int64
    )

    return DocumentGraph(
        doc_id=document.doc_id,
        texts=[w.text for w in words],
        features=features,
        edge_index=edge_index,
        edge_type=edge_type,
        labels=labels,
        tags=tags,
        pages=pages,
        stats=stats,
    )


def collect_fieldtypes(documents: list[Document]) -> list[str]:
    """Every field type present, sorted, for building a stable label vocabulary."""
    seen: set[str] = set()
    for document in documents:
        seen |= document.fieldtypes
    return sorted(seen)
