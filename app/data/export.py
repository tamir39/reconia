"""Export any dataset to a portable, sharded intermediate format.

Every model in the comparison reads this one export. The graph builder turns it into
graphs; a tokeniser turns it into token sequences for LayoutLMv3 or RoBERTa. That is what
makes the comparison fair: each model provably sees the same words, the same boxes and the
same labels, rather than each reading the raw dataset in its own way.

It is also what makes training on Kaggle practical. A dataset of 5,680 separate files
uploads and reads slowly there; a handful of compressed shards does not.

Layout produced::

    <out_dir>/
        manifest.json          counts, field types, shard list, checksums
        train-000.jsonl.gz     one document per line
        train-001.jsonl.gz
        val-000.jsonl.gz

One JSON object per line keeps the export inspectable with ordinary tools and portable
across Python versions, which a pickle or a framework-specific format would not be.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from app.data.schema import BBox, DatasetAdapter, Document, Field, Word

EXPORT_VERSION = 1
DEFAULT_SHARDS = 20


def _round(value: float, places: int = 5) -> float:
    """Trim coordinate precision.

    Five decimal places on a normalised coordinate is well under one pixel on any page
    size we handle, and it makes the export noticeably smaller.
    """
    return round(float(value), places)


def document_to_dict(document: Document) -> dict:
    return {
        "doc_id": document.doc_id,
        "source": document.source,
        "page_count": document.page_count,
        "words": [
            {
                "t": w.text,
                "b": [_round(w.bbox.x0), _round(w.bbox.y0), _round(w.bbox.x1), _round(w.bbox.y1)],
                "p": w.page,
                **({"c": _round(w.confidence, 4)} if w.confidence is not None else {}),
            }
            for w in document.words
        ],
        "fields": [
            {
                "f": f.fieldtype,
                "b": [_round(f.bbox.x0), _round(f.bbox.y0), _round(f.bbox.x1), _round(f.bbox.y1)],
                "p": f.page,
                **({"x": f.text} if f.text is not None else {}),
                **({"li": f.line_item_id} if f.line_item_id is not None else {}),
            }
            for f in document.fields
        ],
    }


def document_from_dict(payload: dict) -> Document:
    words = [
        Word(
            text=w["t"],
            bbox=BBox(*w["b"]),
            page=w.get("p", 0),
            confidence=w.get("c"),
        )
        for w in payload["words"]
    ]
    fields = [
        Field(
            fieldtype=f["f"],
            bbox=BBox(*f["b"]),
            page=f.get("p", 0),
            text=f.get("x"),
            line_item_id=f.get("li"),
        )
        for f in payload["fields"]
    ]
    return Document(
        doc_id=payload["doc_id"],
        words=words,
        fields=fields,
        page_count=payload.get("page_count", 1),
        source=payload.get("source", ""),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class SplitExport:
    split: str
    documents: int
    words: int
    fields: int
    shards: list[str]


def write_shards(
    documents: Iterable[Document],
    out_dir: Path,
    split: str,
    shards: int = DEFAULT_SHARDS,
) -> SplitExport:
    if shards < 1:
        raise ValueError("shards must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        gzip.open(out_dir / f"{split}-{i:03d}.jsonl.gz", "wt", encoding="utf-8")
        for i in range(shards)
    ]
    counts = SplitExport(split=split, documents=0, words=0, fields=0, shards=[])
    try:
        for index, document in enumerate(documents):
            # Round-robin rather than contiguous blocks, so every shard holds a
            # representative slice and a partial download is still usable.
            handles[index % shards].write(
                json.dumps(document_to_dict(document), ensure_ascii=False) + "\n"
            )
            counts.documents += 1
            counts.words += len(document.words)
            counts.fields += len(document.fields)
    finally:
        for handle in handles:
            handle.close()

    counts.shards = [f"{split}-{i:03d}.jsonl.gz" for i in range(shards)]
    return counts


def read_shards(directory: str | Path, split: str) -> Iterator[Document]:
    """Stream documents back. Shards are read in order, documents in shard order."""
    directory = Path(directory)
    for path in sorted(directory.glob(f"{split}-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield document_from_dict(json.loads(line))


def export_dataset(
    adapter: DatasetAdapter,
    out_dir: str | Path,
    splits: Iterable[str] = ("train", "val"),
    shards: int = DEFAULT_SHARDS,
) -> dict:
    """Write every split and a manifest describing the result."""
    splits = tuple(splits)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = set(adapter.splits())
    source_splits = {
        s: ("validation" if s == "val" and s not in available and "validation" in available else s)
        for s in splits
    }
    missing = [s for s in splits if source_splits[s] not in available]
    if missing:
        raise ValueError(
            f"{adapter.name} has no split(s) {missing}; available: {sorted(available)}"
        )

    exports: list[SplitExport] = []
    fieldtypes: set[str] = set()

    for split in splits:
        def counted() -> Iterator[Document]:
            for document in adapter.iter_documents(source_splits[split]):
                fieldtypes.update(document.fieldtypes)
                yield document

        exports.append(write_shards(counted(), out_dir, split, shards))

    manifest = {
        "version": EXPORT_VERSION,
        "dataset": adapter.name,
        "fieldtypes": sorted(fieldtypes),
        "splits": {
            e.split: {
                "documents": e.documents,
                "words": e.words,
                "fields": e.fields,
                "shards": e.shards,
            }
            for e in exports
        },
        "checksums": {
            name: _sha256(out_dir / name) for e in exports for name in e.shards
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def load_manifest(directory: str | Path) -> dict:
    with (Path(directory) / "manifest.json").open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("version") != EXPORT_VERSION:
        raise ValueError(
            f"export was written by version {manifest.get('version')}, "
            f"expected {EXPORT_VERSION}"
        )
    return manifest


def verify(directory: str | Path) -> list[str]:
    """Re-check every shard against the manifest. Returns the names that failed.

    Worth running after transferring the export to Kaggle: a silently truncated upload
    would otherwise show up as unexplained accuracy loss rather than as an error.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    failures: list[str] = []
    for name, expected in manifest["checksums"].items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected:
            failures.append(name)
    return failures


def main(argv: list[str] | None = None) -> int:
    import argparse
    from app.data import CordAdapter, DocileAdapter, SroieAdapter

    adapters = {"docile": DocileAdapter, "cord": CordAdapter, "sroie": SroieAdapter}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(adapters))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    args = parser.parse_args(argv)
    manifest = export_dataset(adapters[args.dataset](), args.output, splits=tuple(args.splits), shards=args.shards)
    failures = verify(args.output)
    if failures:
        raise ValueError(f"Export checksum verification failed: {failures}")
    print(json.dumps({"dataset": manifest["dataset"], "documents": {
        split: data["documents"] for split, data in manifest["splits"].items()
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
