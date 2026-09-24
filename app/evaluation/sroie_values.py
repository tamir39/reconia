"""Score SROIE original key values consistently for OCR models and VLMs.

OCR fragments must be supplied in OCR reading order. A VLM supplies one text per
key. Both pass through the same aggregation and normalization; no confidence or
ground-truth-driven fragment selection is allowed.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from app.evaluation.values import compute_value_metrics, normalize_value

FIELDS = {"company", "date", "address", "total"}
PROTOCOL = "sroie-four-field-nfkc-casefold-v1"


def aggregate_fields(predictions):
    aggregated = {}
    for doc_id, fields in predictions.items():
        grouped = defaultdict(list)
        for field in fields:
            name, text = field["fieldtype"], field.get("text")
            if name not in FIELDS:
                raise ValueError(f"Unknown SROIE field: {name}")
            if text is not None and not isinstance(text, str):
                raise ValueError("SROIE values must be strings or null")
            if normalize_value(text):
                grouped[name].append(text.strip())
        aggregated[doc_id] = [
            {"fieldtype": name, "text": " ".join(parts)}
            for name, parts in grouped.items()
        ]
    return aggregated


def compute_sroie_metrics(predictions, raw_targets):
    if predictions.keys() != raw_targets.keys():
        raise ValueError("Prediction and target document IDs must match exactly")
    aggregated = aggregate_fields(predictions)
    scored = {}
    missing_annotations = 0
    for doc_id, targets in raw_targets.items():
        names = [field["fieldtype"] for field in targets]
        if len(names) != len(set(names)) or not set(names) <= FIELDS:
            raise ValueError("Raw SROIE targets must have one value per known key")
        if any(not isinstance(field.get("text"), str) or not normalize_value(field["text"])
               for field in targets):
            raise ValueError("Annotated SROIE values must be nonempty strings")
        # A missing source annotation is unknown, not a negative example.
        # Preserve all predictions in the output; mask only for scoring.
        missing_annotations += len(FIELDS - set(names))
        scored[doc_id] = [f for f in aggregated[doc_id] if f["fieldtype"] in names]
    result = compute_value_metrics(scored, raw_targets)
    result.update(protocol=PROTOCOL,
                  aggregation="join-nonempty-fragments-in-input-order",
                  target_source="original-SROIE-key-annotations",
                  unannotated_field_slots=missing_annotations,
                  missing_annotation_policy="exclude-unannotated-keys-from-scoring")
    return result, aggregated


def score_files(predictions_path, targets_path, split, output, limit=None):
    predictions_bytes = predictions_path.read_bytes()
    targets_bytes = targets_path.read_bytes()
    targets = json.loads(targets_bytes)[split]
    expected_count = len(targets)
    if limit is not None:
        if limit <= 0:
            raise ValueError("Limit must be positive")
        targets = dict(list(targets.items())[:limit])
    result, aggregated = compute_sroie_metrics(json.loads(predictions_bytes), targets)
    result.update(split=split, expected_num_documents=expected_count,
                  complete=len(targets) == expected_count,
                  prediction_sha256=hashlib.sha256(predictions_bytes).hexdigest(),
                  raw_targets_sha256=hashlib.sha256(targets_bytes).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    values_path = output.with_suffix(".values.json")
    values_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    result["aggregated_prediction_sha256"] = hashlib.sha256(values_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    metrics = score_files(args.predictions, args.targets, args.split, args.output, args.limit)
    print(json.dumps({k: metrics[k] for k in ("protocol", "num_documents", "complete", "f1")}, indent=2))
