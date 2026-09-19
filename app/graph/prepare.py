"""Build the label vocabulary and warm the graph cache for a dataset.

Run once per dataset before training:

    python -m app.graph.prepare --dataset docile --splits train val

The vocabulary is always derived from the training split alone, even when other splits are
being cached. Deriving it from validation as well would leak information about which field
types occur there.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.data import CordAdapter, DocileAdapter, SroieAdapter
from app.data.schema import DatasetAdapter
from app.graph.cache import BuildConfig, GraphCache, build_vocabulary, save_vocabulary
from app.graph.labels import LabelStats

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache"

ADAPTERS: dict[str, type] = {
    "docile": DocileAdapter,
    "cord": CordAdapter,
    "sroie": SroieAdapter,
}


def get_adapter(name: str, root: str | None) -> DatasetAdapter:
    try:
        factory = ADAPTERS[name]
    except KeyError:
        raise SystemExit(
            f"unknown dataset {name!r}; available: {', '.join(sorted(ADAPTERS))}"
        ) from None
    return factory(root) if root else factory()


def prepare(
    adapter: DatasetAdapter,
    cache_dir: Path,
    splits: list[str],
    config: BuildConfig,
    *,
    limit: int | None = None,
    rebuild: bool = False,
) -> None:
    vocabulary_path = cache_dir / f"{adapter.name}-vocabulary.json"

    print(f"[{adapter.name}] building vocabulary from the full train split...")
    started = time.time()
    train_documents = list(adapter.iter_documents("train"))
    vocabulary, fieldtypes = build_vocabulary(train_documents)
    save_vocabulary(vocabulary_path, vocabulary, fieldtypes)
    print(
        f"  {len(fieldtypes)} field types, {len(vocabulary)} BIO labels, "
        f"{len(train_documents)} documents read in {time.time() - started:.1f}s"
    )
    print(f"  saved to {vocabulary_path}")

    cache = GraphCache(cache_dir, vocabulary, config)
    print(f"  cache fingerprint: {config.fingerprint()}")

    for split in splits:
        documents = (
            train_documents
            if split == "train" and limit is None
            else list(adapter.iter_documents(split, limit=limit))
        )
        if limit is not None:
            documents = documents[:limit]

        print(f"\n[{adapter.name}/{split}] caching {len(documents)} documents...")
        started = time.time()
        aggregate = LabelStats()
        nodes = edges = 0

        for index, document in enumerate(documents, start=1):
            graph = cache.get(document, rebuild=rebuild)
            aggregate.merge(graph.stats)
            nodes += graph.num_nodes
            edges += graph.num_edges
            if index % 250 == 0 or index == len(documents):
                elapsed = time.time() - started
                print(
                    f"  {index}/{len(documents)}  "
                    f"{elapsed:.0f}s  ({1000 * elapsed / index:.0f} ms/doc)"
                )

        count = max(len(documents), 1)
        print(f"  nodes/doc {nodes / count:.0f}  edges/doc {edges / count:.0f}")
        print(
            f"  label coverage {100 * aggregate.coverage:.2f}%  "
            f"({aggregate.matched_fields}/{aggregate.total_fields} fields, "
            f"{aggregate.unmatched_fields} unreachable)"
        )
        if aggregate.matched_fields:
            share = 100 * aggregate.multiword_fields / aggregate.matched_fields
            print(f"  multi-word fields {share:.1f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="docile", choices=sorted(ADAPTERS))
    parser.add_argument("--root", default=None, help="dataset directory override")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, default=None, help="documents per split")
    parser.add_argument("--rebuild", action="store_true", help="ignore cached graphs")
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument("--no-same-line", action="store_true")
    parser.add_argument("--no-vertical", action="store_true")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args(argv)

    config = BuildConfig(
        use_knn=not args.no_knn,
        use_same_line=not args.no_same_line,
        use_vertical=not args.no_vertical,
        k=args.k,
    )
    prepare(
        get_adapter(args.dataset, args.root),
        Path(args.cache),
        args.splits,
        config,
        limit=args.limit,
        rebuild=args.rebuild,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
