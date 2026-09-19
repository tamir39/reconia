"""Edge construction strategies.

Nodes are OCR words; edges encode the spatial relationships that carry meaning in a
document. A total on an invoice is identified by sitting at the end of a row and below a
column of amounts, not by its own text, so the edge set is what lets the model use layout.

Three strategies are provided and can be combined. Each returns undirected pairs; the
builder deduplicates and records which strategy produced each edge, so their individual
contribution can be measured by ablation.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from app.data.schema import Word

# Edge type ids, stored per edge so a model can treat relations differently.
EDGE_KNN = 0
EDGE_SAME_LINE = 1
EDGE_VERTICAL = 2

EDGE_TYPE_NAMES = {EDGE_KNN: "knn", EDGE_SAME_LINE: "same_line", EDGE_VERTICAL: "vertical"}


def _centers(words: list[Word]) -> list[tuple[float, float]]:
    return [w.bbox.center for w in words]


def knn_edges(words: list[Word], k: int = 6) -> list[tuple[int, int]]:
    """Connect each word to its `k` nearest neighbours on the same page.

    Provides baseline connectivity so no node is isolated on its page. Distance is computed on
    normalised coordinates, so it is resolution independent.
    """
    n = len(words)
    if n < 2:
        return []

    by_page: dict[int, list[int]] = {}
    for i, w in enumerate(words):
        by_page.setdefault(w.page, []).append(i)

    centers = _centers(words)
    edges: set[tuple[int, int]] = set()
    for indices in by_page.values():
        m = len(indices)
        if m < 2:
            continue
        page_k = min(k, m - 1)
        for i in indices:
            xi, yi = centers[i]
            distances = sorted(
                (
                    (math.hypot(xi - centers[j][0], yi - centers[j][1]), j)
                    for j in indices
                    if j != i
                ),
                key=lambda pair: pair[0],
            )
            for _, j in distances[:page_k]:
                edges.add((i, j) if i < j else (j, i))
    return sorted(edges)


def same_line_edges(
    words: list[Word], overlap_threshold: float = 0.5, max_gap: float = 0.05
) -> list[tuple[int, int]]:
    """Connect words that sit on the same visual line and are horizontally adjacent.

    Only adjacent pairs are connected, not every pair on the line: chaining neighbours
    keeps the graph sparse while still letting information travel along the line through
    successive message passing steps.
    """
    n = len(words)
    if n < 2:
        return []

    by_page: dict[int, list[int]] = {}
    for i, w in enumerate(words):
        by_page.setdefault(w.page, []).append(i)

    edges: set[tuple[int, int]] = set()
    for indices in by_page.values():
        lines: list[list[int]] = []
        for idx in sorted(indices, key=lambda i: (words[i].bbox.y0, words[i].bbox.x0)):
            for line in lines:
                if words[idx].bbox.vertical_overlap(words[line[0]].bbox) >= overlap_threshold:
                    line.append(idx)
                    break
            else:
                lines.append([idx])

        for line in lines:
            ordered = sorted(line, key=lambda i: words[i].bbox.x0)
            for a, b in zip(ordered, ordered[1:]):
                gap = words[b].bbox.x0 - words[a].bbox.x1
                if gap <= max_gap:
                    edges.add((a, b) if a < b else (b, a))
    return sorted(edges)


def vertical_edges(
    words: list[Word], overlap_threshold: float = 0.3, max_gap: float = 0.08
) -> list[tuple[int, int]]:
    """Connect each word to the nearest word directly above and below it.

    This is what carries column structure: a value under a header, or an amount in the
    same column as a total.
    """
    n = len(words)
    if n < 2:
        return []

    by_page: dict[int, list[int]] = {}
    for i, w in enumerate(words):
        by_page.setdefault(w.page, []).append(i)

    edges: set[tuple[int, int]] = set()
    for indices in by_page.values():
        for i in indices:
            below: tuple[float, int] | None = None
            for j in indices:
                if i == j:
                    continue
                gap = words[j].bbox.y0 - words[i].bbox.y1
                if gap < 0 or gap > max_gap:
                    continue
                if words[i].bbox.horizontal_overlap(words[j].bbox) < overlap_threshold:
                    continue
                if below is None or gap < below[0]:
                    below = (gap, j)
            if below is not None:
                j = below[1]
                edges.add((i, j) if i < j else (j, i))
    return sorted(edges)


def build_edges(
    words: list[Word],
    *,
    use_knn: bool = True,
    use_same_line: bool = True,
    use_vertical: bool = True,
    k: int = 6,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Combine the enabled strategies.

    Returns deduplicated pairs and a matching list of edge types. When several strategies
    produce the same pair, the more specific relation is kept: same-line and vertical
    describe document structure, while k-nearest is only proximity.
    """
    typed: dict[tuple[int, int], int] = {}

    def add(pairs: Iterable[tuple[int, int]], edge_type: int, *, overwrite: bool) -> None:
        for pair in pairs:
            if overwrite or pair not in typed:
                typed[pair] = edge_type

    if use_knn:
        add(knn_edges(words, k=k), EDGE_KNN, overwrite=False)
    if use_same_line:
        add(same_line_edges(words), EDGE_SAME_LINE, overwrite=True)
    if use_vertical:
        add(vertical_edges(words), EDGE_VERTICAL, overwrite=True)

    pairs = sorted(typed)
    return pairs, [typed[p] for p in pairs]


def compute_spatial_edge_features(
    node_features: np.ndarray, edge_index: np.ndarray
) -> np.ndarray:
    """Compute continuous spatial relations r_ij between connected nodes.

    Follows Liu et al. (2019) & PICK (2020):
    r_ij = [dx, dy, w_i/h_i, h_j/h_i, w_j/h_i, dist_ij]
    where node_features contains [x0, y0, x1, y1, w, h, center_x, center_y, ...].
    """
    if edge_index.shape[1] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    src, dst = edge_index[0], edge_index[1]
    # Indices from FEATURE_NAMES in builder.py:
    # 4: width, 5: height, 6: center_x, 7: center_y
    dx = node_features[dst, 6] - node_features[src, 6]
    dy = node_features[dst, 7] - node_features[src, 7]
    wi = node_features[src, 4]
    hi = np.maximum(node_features[src, 5], 1e-4)
    wj = node_features[dst, 4]
    hj = node_features[dst, 5]

    aspect_i = wi / hi
    rel_h = hj / hi
    rel_w = wj / hi
    dist = np.sqrt(dx * dx + dy * dy)

    return np.stack([dx, dy, aspect_i, rel_h, rel_w, dist], axis=1).astype(np.float32)
