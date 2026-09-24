"""Fetch pinned receipt sources on Kaggle and match the reviewed local inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.prepare_receipts import (
    CORD_REPO, CORD_REVISION, SROIE_URL, SROIE_SHA256,
    prepare_sroie, receipt_files, sha256, write_json,
)


def receipt_digest(path):
    # This generated metadata is CRLF on Windows and LF on Linux.
    # Images, OCR and key annotation files must still match byte-for-byte.
    if path.name == "source-id-map.json":
        content = json.dumps(json.loads(path.read_text()), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()
    return sha256(path)


def verify_cord(dataset, expected):
    from datasets import Image

    if set(dataset) != set(expected):
        raise ValueError("CORD split names differ from reviewed inputs")
    for split, records in expected.items():
        if len(dataset[split]) != len(records):
            raise ValueError(f"CORD split size differs: {split}")
        for index, row in enumerate(dataset[split].cast_column("image", Image(decode=False))):
            image_bytes = row["image"]["bytes"]
            if image_bytes is None:
                raise ValueError(f"Missing embedded CORD image: {split}/{index}")
            gt = row["ground_truth"]
            if isinstance(gt, str):
                gt = json.loads(gt)
            actual = {
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "ground_truth_sha256": hashlib.sha256(
                    json.dumps(gt, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            }
            if actual != records[index]:
                raise ValueError(f"CORD receipt differs from reviewed inputs: {split}/{index}")


def fetch(dataset_name, destination, reference_path, targets_path):
    reference = json.loads(reference_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / dataset_name
    if target.exists():
        raise ValueError("Use a fresh destination for each receipt run")
    if dataset_name == "cord":
        from datasets import load_dataset, load_from_disk

        expected = reference["cord"]
        if expected["repo"] != CORD_REPO or expected["revision"] != CORD_REVISION:
            raise ValueError("Unexpected CORD source revision")
        data = load_dataset(CORD_REPO, revision=CORD_REVISION,
                            cache_dir=str(destination / "hf-cache"))
        data.save_to_disk(str(target))
        # Check saved bytes and row order, not serializer-dependent Arrow hashes.
        verify_cord(load_from_disk(str(target)), expected["splits"])
    elif dataset_name == "sroie":
        import requests

        expected = reference["sroie"]
        if expected["url"] != SROIE_URL or expected["archive_sha256"] != SROIE_SHA256:
            raise ValueError("Unexpected SROIE source")
        archive = destination / "SROIE2019.zip"
        with requests.get(SROIE_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        prepare_sroie(destination)  # Checks pinned archive hash before extraction.
        actual = {p.relative_to(target).as_posix(): receipt_digest(p)
                  for p in target.rglob("*") if p.is_file()}
        if actual != expected["files_sha256"]:
            raise ValueError("SROIE files differ from reviewed local inputs")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    shutil.copyfile(targets_path, target / "field-value-targets.json")
    manifest = {p.relative_to(destination).as_posix(): sha256(p)
                for p in receipt_files(destination, dataset_name)}
    write_json(destination / "checksums.json", manifest)
    print(f"Verified {dataset_name}: all images and labels match reviewed inputs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sroie", "cord"], required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.dataset, args.destination, args.reference, args.targets)
