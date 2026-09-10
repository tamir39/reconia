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

from app.data.schema import Word

# Edge type ids, stored per edge so a model can treat relations differently.
EDGE_KNN = 0
EDGE_SAME_LINE = 1
EDGE_VERTICAL = 2

EDGE_TYPE_NAMES = {EDGE_KNN: "knn", EDGE_SAME_LINE: "same_line", EDGE_VERTICAL: "vertical"}


def _centers(words: list[Word]) -> list[tuple[float, float]]:
    return [w.bbox.center for w in words]


def knn_edges(words: list[Word], k: int = 6) -> list[tuple[int, int]]:
    """Connect each word to its `k` nearest neighbours by box centre distance.

    Provides baseline connectivity so no node is isolated. Distance is computed on
    normalised coordinates, so it is resolution independent.
    """
    n = len(words)
    if n < 2:
        return []

    centers = _centers(words)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        xi, yi = centers[i]
        distances = sorted(
            (
                (math.hypot(xi - centers[j][0], yi - centers[j][1]), j)
                for j in range(n)
                if j != i
            ),
            key=lambda pair: pair[0],
        )
        for _, j in distances[:k]:
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
