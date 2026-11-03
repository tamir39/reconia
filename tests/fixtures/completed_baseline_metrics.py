# Frozen from the completed RoBERTa run; provenance hashes in prepare/completed_baseline_manifest.json.
import re

def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def evaluate_field(gt: list[str], pred: list[str]) -> tuple[int, int, int]:
    gt_items = [norm for text in gt if (norm := normalize_text(text))]
    pred_items = [norm for text in pred if (norm := normalize_text(text))]

    tp = 0
    matched_indices = set()

    for pred_item in pred_items:
        for idx, gt_item in enumerate(gt_items):
            if idx not in matched_indices and pred_item == gt_item:
                tp += 1
                matched_indices.add(idx)
                break

    fp = len(pred_items) - tp
    fn = len(gt_items) - tp
    return tp, fp, fn


def evaluate_sample(
    gt: dict[str, list[str]],
    pred: dict[str, list[str]],
) -> dict[str, tuple[int, int, int]]:
    results: dict[str, tuple[int, int, int]] = {}
    for field in gt.keys() | pred.keys():
        results[field] = evaluate_field(gt.get(field, []), pred.get(field, []))
    return results


def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def evaluate_dataset(
    gts: list[dict[str, list[str]]],
    preds: list[dict[str, list[str]]],
) -> dict[str, float]:
    field_counts: dict[str, list[int]] = {}

    for gt, pred in zip(gts, preds):
        for field, (tp, fp, fn) in evaluate_sample(gt, pred).items():
            if counts := field_counts.get(field):
                counts[0] += tp
                counts[1] += fp
                counts[2] += fn
            else:
                field_counts[field] = [tp, fp, fn]

    total_p = total_r = total_f1 = 0.0

    for tp, fp, fn in field_counts.values():
        p, r, f1 = compute_metrics(tp, fp, fn)
        total_p += p
        total_r += r
        total_f1 += f1

    num_fields = len(field_counts)

    return {
        "precision": total_p / num_fields,
        "recall": total_r / num_fields,
        "f1": total_f1 / num_fields,
    }