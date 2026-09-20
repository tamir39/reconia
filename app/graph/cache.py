"""Persist built graphs and the label vocabulary.

Building a graph costs roughly 100 ms, so a pass over the full DocILE training split takes
about ten minutes. Training runs many epochs over the same documents, so graphs are built
once and reloaded from disk afterwards.

The label vocabulary is persisted for a separate reason. It must be derived from the whole
training split, not from a sample: a type absent from the vocabulary silently becomes `O`,
which would make validation metrics wrong without any error being raised. Saving it also
pins the label ordering, so a checkpoint trained today stays interpretable later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from app.data.schema import Document
from app.graph.builder import DocumentGraph, build_graph
from app.graph.labels import LabelStats

CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Graph construction settings.

    Recorded with the cache so a cache built under different settings is never silently
    reused: changing the edge strategy changes the graph, and reusing stale graphs would
    quietly invalidate any comparison between configurations.
    """

    include_line_items: bool = True
    use_knn: bool = True
    use_same_line: bool = True
    use_vertical: bool = True
    k: int = 6

    def fingerprint(self) -> str:
        parts = [
            f"li{int(self.include_line_items)}",
            f"knn{self.k if self.use_knn else 0}",
            f"line{int(self.use_same_line)}",
            f"vert{int(self.use_vertical)}",
        ]
        return "-".join(parts)


def save_vocabulary(path: Path, vocabulary: dict[str, int], fieldtypes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "fieldtypes": fieldtypes,
        "vocabulary": vocabulary,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_vocabulary(path: Path) -> tuple[dict[str, int], list[str]]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(
            f"vocabulary at {path} was written by version {payload.get('version')}, "
            f"expected {CACHE_VERSION}"
        )
    return payload["vocabulary"], payload["fieldtypes"]


def build_vocabulary(documents: Iterable[Document]) -> tuple[dict[str, int], list[str]]:
    """Derive the BIO vocabulary from documents.

    Pass the complete training split. Building from a sample produces a vocabulary that
    silently drops the rarest field types, which degrades evaluation metrics.
    """
    from app.graph.labels import build_label_vocabulary

    seen: set[str] = set()
    for document in documents:
        seen |= document.fieldtypes
    fieldtypes = sorted(seen)
    return build_label_vocabulary(fieldtypes), fieldtypes


def _graph_path(cache_dir: Path, config: BuildConfig, doc_id: str) -> Path:
    return cache_dir / config.fingerprint() / f"{doc_id}.npz"


def save_graph(path: Path, graph: DocumentGraph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        doc_id=np.asarray(graph.doc_id),
        texts=np.asarray(graph.texts, dtype=object),
        features=graph.features,
        edge_index=graph.edge_index,
        edge_type=graph.edge_type,
        labels=graph.labels,
        tags=np.asarray(graph.tags, dtype=object),
        pages=graph.pages,
        stats=np.asarray(
            [
                graph.stats.total_fields,
                graph.stats.matched_fields,
                graph.stats.unmatched_fields,
                graph.stats.total_words,
                graph.stats.labelled_words,
                graph.stats.multiword_fields,
            ],
            dtype=np.int64,
        ),
    )


def load_graph(path: Path) -> DocumentGraph:
    with np.load(path, allow_pickle=True) as data:
        counts = data["stats"].tolist()
        stats = LabelStats(
            total_fields=counts[0],
            matched_fields=counts[1],
            unmatched_fields=counts[2],
            total_words=counts[3],
            labelled_words=counts[4],
            multiword_fields=counts[5],
        )
        return DocumentGraph(
            doc_id=str(data["doc_id"]),
            texts=[str(t) for t in data["texts"]],
            features=data["features"],
            edge_index=data["edge_index"],
            edge_type=data["edge_type"],
            labels=data["labels"],
            tags=[str(t) for t in data["tags"]],
            pages=data["pages"],
            stats=stats,
        )


class GraphCache:
    """Builds graphs on first use and reads them from disk thereafter."""

    def __init__(
        self,
        cache_dir: str | Path,
        vocabulary: dict[str, int],
        config: BuildConfig | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.vocabulary = vocabulary
        self.config = config or BuildConfig()

    def path_for(self, doc_id: str) -> Path:
        return _graph_path(self.cache_dir, self.config, doc_id)

    def get(self, document: Document, *, rebuild: bool = False) -> DocumentGraph:
        path = self.path_for(document.doc_id)
        if path.is_file() and not rebuild:
            return load_graph(path)
        graph = build_graph(
            document,
            self.vocabulary,
            include_line_items=self.config.include_line_items,
            use_knn=self.config.use_knn,
            use_same_line=self.config.use_same_line,
            use_vertical=self.config.use_vertical,
            k=self.config.k,
        )
        save_graph(path, graph)
        return graph

    def warm(
        self, documents: Iterable[Document], *, rebuild: bool = False
    ) -> Iterator[DocumentGraph]:
        """Build and cache a whole split, yielding each graph as it is produced."""
        for document in documents:
            yield self.get(document, rebuild=rebuild)

    def cached_ids(self) -> set[str]:
        directory = self.cache_dir / self.config.fingerprint()
        if not directory.is_dir():
            return set()
        return {p.stem for p in directory.glob("*.npz")}
