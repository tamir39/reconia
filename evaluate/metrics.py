from collections import defaultdict
import re


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def match_field(pred: list[str] | str | None, gt: list[str] | str | None) -> tuple[int, int, int]:
    if isinstance(gt, list):
        if isinstance(pred, list):
            pred_items = [norm for x in pred if (norm := normalize_text(x))]
        elif pred is not None and (norm_single := normalize_text(pred)):
            pred_items = [norm_single]
        else:
            pred_items = []

        gt_items = [norm for x in gt if (norm := normalize_text(x))]

        tp = 0
        matched_gt = set()
        for p in pred_items:
            for idx, g in enumerate(gt_items):
                if idx not in matched_gt and p == g:
                    tp += 1
                    matched_gt.add(idx)
                    break

        fp = len(pred_items) - tp
        fn = len(gt_items) - tp
        return tp, fp, fn

    if isinstance(pred, list):
        norm_pred = normalize_text(" ".join(str(x) for x in pred if x is not None))
    else:
        norm_pred = normalize_text(pred)

    norm_gt = normalize_text(gt)

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
    gt: dict[str, list[str] | str],
) -> dict[str, tuple[int, int, int]]:
    all_keys = set(gt.keys()) | set(pred.keys())
    results = {}

    for key in all_keys:
        if key == "O":
            continue
        results[key] = match_field(pred.get(key), gt.get(key))

    return results


def compute_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def evaluate_dataset(
    preds: list[dict[str, list[str] | str]],
    gts: list[dict[str, list[str] | str]],
) -> dict[str, float]:
    if len(preds) != len(gts):
        raise ValueError(f"preds ({len(preds)}) and gts ({len(gts)}) length mismatch.")

    field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for pred, gt in zip(preds, gts):
        for field, (tp, fp, fn) in evaluate_sample(pred, gt).items():
            counts = field_counts[field]
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn

    if not field_counts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    total_p = 0.0
    total_r = 0.0
    total_f1 = 0.0

    for counts in field_counts.values():
        scores = compute_metrics(counts[0], counts[1], counts[2])
        total_p += scores["precision"]
        total_r += scores["recall"]
        total_f1 += scores["f1"]

    num_fields = len(field_counts)

    return {
        "precision": round(total_p / num_fields, 4),
        "recall": round(total_r / num_fields, 4),
        "f1": round(total_f1 / num_fields, 4),
    }