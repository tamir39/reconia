"""Data and label contract of the three completed October 24 runs."""

import hashlib
import json
import re
from pathlib import Path


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "prepare/completed_baseline_manifest.json"


def load_manifest(manifest_path=MANIFEST_PATH):
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_labels(dataset_name):
    return load_manifest()["datasets"][dataset_name]["labels"]


def canonical_records(samples):
    records = []
    for sample in samples:
        sample = dict(sample)
        path = sample["image_path"].replace("\\", "/")
        # CORD images can be extracted from Arrow into another absolute cache root.
        match = re.search(r"(?:^|/)(images/(?:train|val|test)/[^/]+)$", path)
        sample["image_path"] = match.group(1) if match else path
        records.append(sample)
    return json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def verify_baseline_data(dataset_name, cache_dir, manifest_path=MANIFEST_PATH):
    """Fail before training if prepared samples, order, annotations or GT change."""
    expected = load_manifest(manifest_path)["datasets"][dataset_name]
    result = {}
    for split, reference in expected["splits"].items():
        path = Path(cache_dir) / f"{split}.jsonl"
        samples = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        digest = hashlib.sha256(canonical_records(samples).encode("utf-8")).hexdigest()
        if len(samples) != reference["count"] or digest != reference["sha256"]:
            raise ValueError(
                f"Prepared {dataset_name}/{split} differs from completed baselines; "
                "check raw data, split order, annotations and GT before training"
            )
        result[split] = {"count": len(samples), "sha256": digest}
    return result
