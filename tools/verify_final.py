"""Read-only checks for the final source, benchmark and optional local weights."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from demo.assets import DATASETS, MODELS, ROOT, checkpoint, file_hash, labels, manifest, sample_image, samples


def canonical_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    content = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    return rows, hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_prepared(datasets=DATASETS):
    from utils.common import get_cache_dir
    contract = json.loads((ROOT / "prepare/input_manifest.json").read_text())
    checked = {}
    for dataset in datasets:
        checked[dataset] = {}
        for split, expected in contract["datasets"][dataset].items():
            rows, digest = canonical_rows(get_cache_dir(dataset) / f"{split}.jsonl")
            if len(rows) != expected["count"] or digest != expected["canonical_sha256"]:
                raise ValueError(f"Prepared {dataset}/{split} differs from the published final split")
            checked[dataset][split] = len(rows)
    return checked


def text_hash(path):
    content = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify(check_weights=False):
    provenance = json.loads((ROOT / "reports/benchmarks/provenance.json").read_text())
    for group in provenance["models"].values():
        for path, info in group["files"].items():
            if text_hash(ROOT / path) != info["final_sha256"]:
                raise ValueError("Source differs from final provenance: " + path)
    for section in ("shared", "preparation"):
        for path, info in provenance[section].items():
            target = info.get("local", path)
            if text_hash(ROOT / target) != info["final_sha256"]:
                raise ValueError("Source differs from final provenance: " + target)
    metadata = manifest()
    examples = json.loads((ROOT / "demo/examples.json").read_text(encoding="utf-8"))
    checked = []
    normalize = lambda values: Counter(" ".join(str(value).lower().split()) for value in values if str(value).strip())
    for model in MODELS:
        for dataset in DATASETS:
            info = metadata["models"][model]["datasets"][dataset]
            base = ROOT / "reports/benchmarks" / dataset / model
            metrics = json.loads((base / "test_metrics.json").read_text())
            if metrics != info["metrics"]:
                raise ValueError(f"Benchmark and demo differ: {model}/{dataset}")
            if not all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in ("precision", "recall", "f1")):
                raise ValueError(f"Invalid metrics: {model}/{dataset}")
            with (base / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            if [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)) or not history:
                raise ValueError(f"History epochs are not consecutive: {model}/{dataset}")
            if not all(math.isfinite(float(value)) for row in history for value in row.values()):
                raise ValueError(f"Nonfinite history: {model}/{dataset}")
            pair = examples[dataset][model]
            if pair["correct"]["id"] == pair["error"]["id"]:
                raise ValueError("The two examples must be different receipts")
            for kind, example in pair.items():
                row = samples(dataset, model)[example["id"]]
                if file_hash(sample_image(dataset, row)) != row["image_sha256"]:
                    raise ValueError("Bundled sample image changed")
                if any("label" in item for item in row["ocr"]):
                    raise ValueError("Ground-truth BIO labels must not enter demo OCR input")
                if normalize(example["truth"]) != normalize(row["gt"].get(example["field"], [])):
                    raise ValueError("Example truth differs from source annotations")
                same = normalize(example["truth"]) == normalize(example["prediction"])
                if same != (kind == "correct") or example["checkpoint_sha256"] != info["checkpoint_sha256"]:
                    raise ValueError("Example outcome or checkpoint provenance differs")
            if not labels(dataset, model) or labels(dataset, model)[0] != "O":
                raise ValueError("Invalid label inventory")
            path = checkpoint(dataset, model)
            if check_weights and (not path.is_file() or path.stat().st_size != info["checkpoint_bytes"] or file_hash(path) != info["checkpoint_sha256"]):
                raise ValueError(f"Missing or changed checkpoint: {model}/{dataset}")
            checked.append({"model": model, "dataset": dataset, "f1": metrics["f1"], "epochs": len(history), "checkpoint_present": path.is_file()})
    extra = json.loads((ROOT / "reports/benchmarks/additional-models.json").read_text())
    published = []
    for model, record in extra["models"].items():
        for dataset, info in record["datasets"].items():
            if info is None:
                continue
            metrics = json.loads((ROOT / info["metrics_file"]).read_text())
            if metrics != info["metrics"] or metrics["model"] != model or metrics["dataset"] != dataset:
                raise ValueError("Published extra result differs from its registry")
            if not all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float))):
                raise ValueError("Nonfinite published extra result")
            if not all(0 <= metrics[k] <= 1 for k in ("precision", "recall", "f1")):
                raise ValueError("Invalid published extra metrics")
            for name, digest in info["output_text_sha256"].items():
                if text_hash((ROOT / info["metrics_file"]).with_name(name)) != digest:
                    raise ValueError("Published extra artifact changed: " + name)
            with (ROOT / info["metrics_file"]).with_name("history.csv").open(newline="") as stream:
                history = list(csv.DictReader(stream))
            if [int(row["epoch"]) for row in history] != list(range(1, info["epochs"] + 1)):
                raise ValueError("Published extra history epochs are not consecutive")
            if not all(math.isfinite(float(v)) for row in history for v in row.values()):
                raise ValueError("Nonfinite published extra history")
            best = max(history, key=lambda row: float(row["f1"]))
            if int(best["epoch"]) != info["best_epoch"] or float(best["f1"]) != info["best_validation_f1"]:
                raise ValueError("Published extra best validation selection differs")
            published.append({"model": model, "dataset": dataset, "f1": metrics["f1"],
                              "inference_rerun": record["inference_rerun"], "checkpoint_downloaded": record["checkpoint_downloaded"]})
    return {"metric": "Micro-F1", "model_dataset_pairs": len(checked), "examples": 2 * len(checked),
            "weights_hashed": check_weights, "checks": checked, "additional_published_results": published,
            "published_result_pairs": len(checked) + len(published)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true", help="Also hash all 15 large checkpoints")
    parser.add_argument("--prepared", action="store_true", help="Verify full published train/val/test splits")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.weights)
    if args.prepared:
        result["prepared"] = verify_prepared()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {result['model_dataset_pairs']} benchmark pairs and {result['examples']} examples (Micro-F1).")
    print(f"Audited {len(result['additional_published_results'])} additional published result(s); total {result['published_result_pairs']} available pairs.")


if __name__ == "__main__":
    main()
