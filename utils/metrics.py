from utils.common import normalize_text


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
    if not gts or len(gts) != len(preds):
        raise ValueError("Expected one prediction per document in a nonempty split")
    total_tp = total_fp = total_fn = 0

    for gt, pred in zip(gts, preds):
        for tp, fp, fn in evaluate_sample(gt, pred).values():
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
