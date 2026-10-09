from collections import defaultdict

from tools.common import normalize_text


def match_field(pred: list[str] | str | None, ground_truth: list[str] | str | None) -> tuple[int, int, int]:
    if isinstance(ground_truth, list):
        if isinstance(pred, list):
            pred_items = [norm for text in pred if (norm := normalize_text(text))]
        elif pred is not None and (norm_single := normalize_text(pred)):
            pred_items = [norm_single]
        else:
            pred_items = []

        gt_items = [norm for text in ground_truth if (norm := normalize_text(text))]

        tp = 0
        matched_gt = set()
        for pred_val in pred_items:
            for idx, gt_val in enumerate(gt_items):
                if idx not in matched_gt and pred_val == gt_val:
                    tp += 1
                    matched_gt.add(idx)
                    break

        fp = len(pred_items) - tp
        fn = len(gt_items) - tp
        return tp, fp, fn

    if isinstance(pred, list):
        norm_pred = normalize_text(" ".join(str(text) for text in pred if text is not None))
    else:
        norm_pred = normalize_text(pred)

    norm_gt = normalize_text(ground_truth)

    if not norm_pred and not norm_gt:
        return 0, 0, 0
    if norm_pred == norm_gt:
        return 1, 0, 0
    if norm_pred and not norm_gt:
        return 0, 1, 0
    if not norm_pred and norm_gt:
        return 0, 0, 1
    return 0, 1, 1


def evaluate_sample(
    pred: dict[str, list[str] | str],
    ground_truth: dict[str, list[str] | str],
) -> dict[str, tuple[int, int, int]]:
    all_keys = set(ground_truth.keys()) | set(pred.keys())
    results = {}

    for key in all_keys:
        if key == "O":
            continue
        results[key] = match_field(pred.get(key), ground_truth.get(key))

    return results


def compute_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_dataset(
    preds: list[dict[str, list[str] | str]],
    ground_truths: list[dict[str, list[str] | str]],
) -> dict[str, float]:
    if len(preds) != len(ground_truths):
        raise ValueError(f"preds ({len(preds)}) and ground_truths ({len(ground_truths)}) length mismatch.")

    field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for pred, ground_truth in zip(preds, ground_truths):
        for field, (tp, fp, fn) in evaluate_sample(pred, ground_truth).items():
            counts = field_counts[field]
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn

    if not field_counts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for counts in field_counts.values():
        scores = compute_metrics(counts[0], counts[1], counts[2])
        total_precision += scores["precision"]
        total_recall += scores["recall"]
        total_f1 += scores["f1"]

    num_fields = len(field_counts)

    return {
        "precision": round(total_precision / num_fields, 4),
        "recall": round(total_recall / num_fields, 4),
        "f1": round(total_f1 / num_fields, 4),
    }