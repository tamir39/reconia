"""Model-independent field-value scoring, including OCR-free model outputs.

This is a local text-value metric, not the official DocILE localization metric.
"""

from collections import Counter
import re
import unicodedata


VALUE_PROTOCOL = "field-value-nfkc-casefold-v1"


def normalize_value(text: str | None) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


def compute_value_metrics(predictions: dict[str, list[dict]], targets: dict[str, list[dict]]) -> dict:
    """Match type + normalized text within each document; preserve duplicate counts."""
    if predictions.keys() != targets.keys():
        raise ValueError("Prediction and target document IDs must match exactly")
    counts: dict[str, Counter] = {}
    for doc_id, fields in targets.items():
        expected = Counter((f["fieldtype"], normalize_value(f.get("text"))) for f in fields)
        actual = Counter((f["fieldtype"], normalize_value(f.get("text"))) for f in predictions[doc_id])
        for key in expected.keys() | actual.keys():
            bucket = counts.setdefault(key[0], Counter())
            tp = min(expected[key], actual[key])
            bucket.update(tp=tp, fp=actual[key] - tp, fn=expected[key] - tp)

    def score(c: Counter) -> dict:
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": tp + fn,
        }

    total = Counter()
    for count in counts.values():
        total.update(count)
    per_type = {key: score(count) for key, count in sorted(counts.items())}
    return {
        "protocol": VALUE_PROTOCOL,
        **score(total),
        "macro_f1": sum(s["f1"] for s in per_type.values()) / len(per_type) if per_type else 0.0,
        "per_type": per_type,
        "num_documents": len(targets),
    }
