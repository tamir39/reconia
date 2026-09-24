"""Prepare pinned SROIE/CORD receipts locally, validate them, and package images.

Run as: python -m scripts.prepare_receipts [--download] [--bundle]
No upload or training is started by this command.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SROIE_URL = "https://github.com/NjoyimPeguy/ICDAR-2019-RRC-SROIE/releases/download/v2.0/SROIE2019.zip"
SROIE_SHA256 = "aebb3e33a5ed0aaf644a5855dee97ca8ae80114e5b43ae96d431ab771c851941"
CORD_REPO = "naver-clova-ix/cord-v2"
CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
EXPECTED = {"sroie": {"train": 500, "val": 63, "test": 63},
            "cord": {"train": 800, "val": 100, "test": 100}}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(cache: Path) -> None:
    import requests
    from datasets import load_dataset

    archive = cache / "SROIE2019.zip"
    if not archive.is_file():
        print("Downloading SROIE release...", flush=True)
        temporary = archive.with_suffix(".part")
        with requests.get(SROIE_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    out.write(chunk)
        with zipfile.ZipFile(temporary) as source:
            if source.testzip() is not None:
                raise ValueError("Corrupt SROIE download")
        temporary.replace(archive)
    cord = cache / "cord"
    if not (cord / "dataset_dict.json").is_file():
        print("Downloading pinned CORD-v2...", flush=True)
        dataset = load_dataset(CORD_REPO, revision=CORD_REVISION, cache_dir=str(cache / "hf"))
        dataset.save_to_disk(str(cord))


def prepare_sroie(cache: Path) -> dict:
    """Keep only canonical receipts, excluding release files named X...(1).txt."""
    source_path = cache / "SROIE2019.zip"
    if sha256(source_path) != SROIE_SHA256:
        raise ValueError("SROIE release differs from the reviewed archive")
    target = cache / "sroie"
    prefixes = {"box": "0325updated.task1train(626p)",
                "key": "0325updated.task2train(626p)",
                "img": "0325updated.task1train(626p)"}
    suffixes = {"box": ".txt", "key": ".txt", "img": ".jpg"}
    with zipfile.ZipFile(source_path) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("Duplicate ZIP member names")
        members = {}
        for kind, prefix in prefixes.items():
            members[kind] = {
                PurePosixPath(name).stem: name for name in source.namelist()
                if PurePosixPath(name).parent.as_posix() == prefix
                and re.fullmatch(r"X[0-9]+" + re.escape(suffixes[kind]), PurePosixPath(name).name)
            }
        ids = set(members["box"])
        if len(ids) != 626 or any(set(part) != ids for part in members.values()):
            raise ValueError("SROIE must contain exactly 626 complete canonical receipts")
        mapping = {f"{i:03d}": original for i, original in enumerate(sorted(ids))}
        for kind in members:
            (target / kind).mkdir(parents=True, exist_ok=True)
        for local_id, original in mapping.items():
            for kind, suffix in {"box": ".csv", "key": ".json", "img": ".jpg"}.items():
                # Construct destination ourselves; never extract archive-supplied paths.
                content = source.read(members[kind][original])
                if kind == "key":
                    key = json.loads(content.decode("utf-8-sig"))
                    if not key or not set(key) <= {"company", "date", "address", "total"}:
                        raise ValueError(f"Unexpected key fields: {original}")
                (target / kind / f"{local_id}{suffix}").write_bytes(content)
    provenance = {
        "url": SROIE_URL, "archive_sha256": sha256(source_path),
        "mapping_policy": "canonical-source-ids-sorted-to-000-625-v1",
        "legacy_mapping_verified": False,
        "note": "Numeric split slots preserved; correspondence to Thinh's older receipts is unverified.",
        "source_ids": mapping,
        "split_sha256": sha256(ROOT / "app/data/sroie_splits.json"),
    }
    write_json(target / "source-id-map.json", provenance)
    shutil.copyfile(ROOT / "app/data/sroie_splits.json", target / "splits.json")
    return provenance


def validate(cache: Path, provenance: dict) -> dict:
    from app.data import CordAdapter, SroieAdapter
    from app.graph.labels import label_document

    report = {"sources": {"sroie": {k: v for k, v in provenance.items() if k != "source_ids"},
                          "cord": {"repo": CORD_REPO, "revision": CORD_REVISION}},
              "datasets": {}, "training_started": False}
    for name, adapter in (("sroie", SroieAdapter(root=cache / "sroie")),
                          ("cord", CordAdapter(root=cache / "cord"))):
        seen_ids, image_splits = set(), {}
        splits = {}
        targets = {}
        for split, expected in EXPECTED[name].items():
            ids = adapter.document_ids(split)
            if len(ids) != expected or len(set(ids)) != expected or seen_ids.intersection(ids):
                raise ValueError(f"Invalid {name}/{split} split")
            seen_ids.update(ids)
            stats = Counter(documents=len(ids))
            targets[split] = {}
            for doc_id in ids:
                # Explicit raw-image checks avoid the adapters' blank-image fallback.
                if name == "sroie" and not adapter.image_path(doc_id).is_file():
                    raise ValueError(f"Missing image: {doc_id}")
                if name == "cord":
                    source_split, index = adapter._parse_doc_id(doc_id)
                    if adapter._dataset[source_split][index].get("image") is None:
                        raise ValueError(f"Missing image: {doc_id}")
                with adapter.get_image(doc_id) as image:
                    image.load()
                    if min(image.size) <= 1:
                        raise ValueError(f"Invalid image: {doc_id}")
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    if digest in image_splits and image_splits[digest] != split:
                        raise ValueError(f"Identical receipt image across {name} splits: {doc_id}")
                    image_splits[digest] = split
                document = adapter.load(doc_id)
                if not document.words:
                    raise ValueError(f"Empty OCR: {doc_id}")
                for word in document.words:
                    b = word.bbox
                    if not (0 <= b.x0 <= b.x1 <= 1 and 0 <= b.y0 <= b.y1 <= 1):
                        raise ValueError(f"Invalid box: {doc_id}")
                tags, matching = label_document(document)
                if len(tags) != len(document.words):
                    raise ValueError(f"BIO alignment failed: {doc_id}")
                stats.update(images=1, words=len(document.words), fields=len(document.fields),
                             documents_without_fields=int(not document.fields))
                if name == "sroie":
                    # Preserve the four original values for future VLM scoring, not OCR fragments.
                    raw = json.loads((cache / "sroie/key" / f"{doc_id}.json").read_text(encoding="utf-8-sig"))
                    stats.update(missing_raw_key_fields=4 - len(raw))
                    targets[split][doc_id] = [{"fieldtype": k, "text": v} for k, v in raw.items()]
                else:
                    targets[split][doc_id] = [{"fieldtype": f.fieldtype, "text": f.text} for f in document.fields]
            splits[split] = dict(stats)
            print(f"Validated {name}/{split}: {stats['documents']} receipts", flush=True)
        write_json(cache / name / "field-value-targets.json", targets)
        report["datasets"][name] = {"splits": splits, "duplicate_images_across_splits": 0}
    report["evaluation_notes"] = [
        "SROIE 500/63/63 is an internal split of the original training receipts, not the official test.",
        "SROIE OCR BIO fields are heuristically aligned fragments; use raw four-field targets and aggregate predictions for VLM comparisons. That scoring integration is still pending.",
        "CORD inputs use annotated transcripts/boxes; describe this as post-OCR parsing, not an end-to-end OCR benchmark.",
    ]
    write_json(cache / "validation.json", report)
    return report


def receipt_files(cache: Path, name: str) -> list[Path]:
    """Reject unreviewed files instead of recursively uploading arbitrary content."""
    root = cache / name
    if name == "sroie":
        mapping = json.loads((root / "source-id-map.json").read_text(encoding="utf-8"))
        ids = mapping["source_ids"]
        if set(ids) != {f"{i:03d}" for i in range(626)}:
            raise ValueError("Unexpected receipt mapping")
        allowed = {root / folder / f"{doc_id}{suffix}" for doc_id in ids
                   for folder, suffix in (("img", ".jpg"), ("box", ".csv"), ("key", ".json"))}
        allowed.update(root / f for f in ("source-id-map.json", "splits.json", "field-value-targets.json"))
    elif name == "cord":
        allowed = {root / "dataset_dict.json", root / "field-value-targets.json"}
        for split in ("train", "validation", "test"):
            allowed.update(root / split / f for f in ("dataset_info.json", "state.json"))
            state = json.loads((root / split / "state.json").read_text(encoding="utf-8"))
            for shard in state["_data_files"]:
                filename = shard["filename"]
                if not re.fullmatch(r"data-\d+-of-\d+\.arrow", filename):
                    raise ValueError("Unexpected CORD shard path")
                allowed.add(root / split / filename)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    actual = {p for p in root.rglob("*") if p.is_file()}
    if actual != allowed:
        raise ValueError(f"Unreviewed or missing files in {name}: "
                         f"{sorted(str(p.relative_to(root)) for p in actual ^ allowed)}")
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in allowed):
        raise ValueError("Dataset files must stay within the dataset directory")
    return sorted(allowed)


def bundle(cache: Path) -> None:
    """Build allowlisted data and source archives, never credentials or weights."""
    out = cache / "kaggle"
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in EXPECTED:
        files = receipt_files(cache, name)
        manifest = {p.relative_to(cache).as_posix(): sha256(p) for p in files}
        archive_path = out / f"{name}-receipts.bin"
        # Kaggle recursively expands .zip files; .bin preserves a verifiable container.
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            for path in files:
                archive.write(path, path.relative_to(cache).as_posix())
            archive.writestr("checksums.json", json.dumps(manifest, indent=2))
        hashes[archive_path.name] = {"bytes": archive_path.stat().st_size, "sha256": sha256(archive_path)}
    write_json(out / "bundle-manifest.json", hashes)
    shutil.copyfile(cache / "validation.json", out / "validation.json")
    sources = sorted(p for p in (ROOT / "app").rglob("*")
                     if p.is_file() and p.suffix in {".py", ".json"})
    provenance = {"base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}}
    with zipfile.ZipFile(out / "reconia-code.bin", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("source-provenance.json", json.dumps(provenance, indent=2))
    hashes["reconia-code.bin"] = {"bytes": (out / "reconia-code.bin").stat().st_size,
                                   "sha256": sha256(out / "reconia-code.bin")}
    write_json(out / "bundle-manifest.json", hashes)
    print(json.dumps(hashes, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/receipt-prep")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(args.cache)
    provenance = prepare_sroie(args.cache)
    validate(args.cache, provenance)
    if args.bundle:
        bundle(args.cache)


if __name__ == "__main__":
    main()
