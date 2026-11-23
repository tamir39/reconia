"""Re-run the 30 bundled test cases with the pinned checkpoints (no training)."""
import json
from pathlib import Path
from time import perf_counter

from PIL import Image

from demo.assets import DATASETS, MODELS, ROOT, sample_image, samples
from demo.inference import predict


def main():
    examples = json.loads((ROOT / "demo/examples.json").read_text(encoding="utf-8"))
    checks = []
    for model in MODELS:
        for dataset in DATASETS:
            for kind, example in examples[dataset][model].items():
                row = samples(dataset, model)[example["id"]]
                start = perf_counter()
                # Only OCR text/boxes enter prediction; GT is solely for the displayed comparison.
                with Image.open(sample_image(dataset, row)) as image:
                    fields = predict(dataset, model, {"ocr": row["ocr"]}, image.convert("RGB"))
                expected = example["fields"] if "fields" in example else example.get("prediction_fields")
                if expected is None:
                    expected = {example["field"]: example["prediction"]}
                    actual = {example["field"]: fields.get(example["field"], [])}
                else:
                    actual = fields
                if actual != expected:
                    raise AssertionError(f"Prediction changed: {model}/{dataset}/{kind}: {actual!r} != {expected!r}")
                checks.append({"model": model, "dataset": dataset, "kind": kind, "id": row["id"],
                               "seconds": round(perf_counter() - start, 2), "matched": True})
                print(f"PASS {model}/{dataset}/{kind}", flush=True)
    target = ROOT / "output/verification/demo-predictions.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"passed": True, "count": len(checks), "checks": checks}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
