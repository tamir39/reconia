"""Evaluation metrics for invoice and document information extraction.
Provides precision, recall, f1, and inference latency/throughput profiling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class EntitySpan:
    """An extracted entity span on a document."""

    entity_type: str
    start_token: int
    end_token: int  # exclusive
    text: str | None = None
    page: int = 0
    confidence: float = 1.0

    @property
    def key(self) -> tuple[str, int, int, int]:
        """Unique span key for exact coordinate matching: (type, page, start, end)."""
        return (self.entity_type, self.page, self.start_token, self.end_token)


def extract_entities_from_bio(
    tags: list[str],
    texts: list[str] | None = None,
    pages: list[int] | np.ndarray | None = None,
) -> list[EntitySpan]:
    """Extract structured EntitySpan objects from a sequence of BIO tags.

    Recognizes standard BIO formats:
      - 'B-field_name' starts a new entity.
      - 'I-field_name' continues the current entity if field_name matches.
      - 'O' indicates outside of any entity.
    """
    entities: list[EntitySpan] = []
    current_type: str | None = None
    start_idx = 0
    current_page = 0

    def get_page(idx: int) -> int:
        if pages is None or idx >= len(pages):
            return 0
        return int(pages[idx])

    def get_text_slice(s: int, e: int) -> str | None:
        if texts is None or s >= len(texts):
            return None
        return " ".join(texts[s:e])

    def flush(end_idx: int) -> None:
        nonlocal current_type, start_idx, current_page
        if current_type is not None:
            entities.append(
                EntitySpan(
                    entity_type=current_type,
                    start_token=start_idx,
                    end_token=end_idx,
                    text=get_text_slice(start_idx, end_idx),
                    page=current_page,
                )
            )
            current_type = None

    for i, tag in enumerate(tags):
        page_i = get_page(i)
        if tag.startswith("B-"):
            flush(i)
            current_type = tag[2:]
            start_idx = i
            current_page = page_i

        elif tag.startswith("I-"):
            tag_type = tag[2:]
            # If I- matches current entity type and is on the same page, continue
            if current_type != tag_type or current_page != page_i:
                # If stray I- tag or cross-page transition, end previous and start new
                flush(i)
                current_type = tag_type
                start_idx = i
                current_page = page_i

        else:  # 'O' or other non-entity tag
            flush(i)

    # Handle entity extending to the end of document
    flush(len(tags))

    return entities


def compute_token_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str] | dict[int, str],
    *,
    ignore_labels: set[str] | None = None,
) -> dict[str, Any]:
    """Compute token-level classification metrics: accuracy, precision, recall, f1, and per-class stats.

    Args:
        y_true: 1D array of ground truth label IDs.
        y_pred: 1D array of predicted label IDs.
        label_names: Mapping from integer ID to string tag name (e.g. ['O', 'B-total', ...]).
        ignore_labels: Tags to exclude from F1 calculation (defaults to {'O'}).
    """
    if ignore_labels is None:
        ignore_labels = {"O"}

    y_t = np.asarray(y_true, dtype=np.int64).flatten()
    y_p = np.asarray(y_pred, dtype=np.int64).flatten()

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    if isinstance(label_names, dict):
        id_to_name = label_names
    else:
        id_to_name = {i: name for i, name in enumerate(label_names)}

    num_classes = max(max(id_to_name.keys(), default=0) + 1, int(y_t.max(initial=0)) + 1)

    # Calculate per-class True Positives, False Positives, False Negatives
    per_class_stats: dict[str, dict[str, float]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    f1_list: list[float] = []
    prec_list: list[float] = []
    rec_list: list[float] = []

    for cls_id, name in sorted(id_to_name.items()):
        is_true = y_t == cls_id
        is_pred = y_p == cls_id

        tp = int(np.sum(is_true & is_pred))
        fp = int(np.sum(~is_true & is_pred))
        fn = int(np.sum(is_true & ~is_pred))
        support = int(np.sum(is_true))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_stats[name] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
        }

        if name not in ignore_labels:
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    overall_accuracy = float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

    return {
        "accuracy": overall_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_tokens": int(len(y_t)),
        "per_class": per_class_stats,
    }


def compute_entity_metrics(
    pred_entities_batch: list[list[EntitySpan]],
    true_entities_batch: list[list[EntitySpan]],
    *,
    match_mode: str = "exact",
) -> dict[str, Any]:
    """Compute entity-level Precision, Recall, and F1 across a batch of documents.

    Args:
        pred_entities_batch: List of extracted EntitySpan lists per document.
        true_entities_batch: List of ground-truth EntitySpan lists per document.
        match_mode: 'exact' (exact span matching: type, page, start_token, end_token)
                    or 'text' (matches entity_type and text content).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_type_tp: dict[str, int] = {}
    per_type_fp: dict[str, int] = {}
    per_type_fn: dict[str, int] = {}
    all_types: set[str] = set()

    if len(pred_entities_batch) != len(true_entities_batch):
        raise ValueError("Predictions and ground truth must contain the same number of documents")

    for pred_list, true_list in zip(pred_entities_batch, true_entities_batch):
        if match_mode == "exact":
            true_keys = [e.key for e in true_list]
            pred_keys = [e.key for e in pred_list]
        elif match_mode == "text":
            true_keys = [(e.entity_type, (e.text or "").strip().lower()) for e in true_list]
            pred_keys = [(e.entity_type, (e.text or "").strip().lower()) for e in pred_list]
        else:
            raise ValueError(f"Unknown match_mode {match_mode!r}; choose 'exact' or 'text'")

        # Register types
        for e in true_list:
            all_types.add(e.entity_type)
        for e in pred_list:
            all_types.add(e.entity_type)

        matched_true_indices = set()
        for p_idx, p_key in enumerate(pred_keys):
            p_type = pred_list[p_idx].entity_type
            match_found = False
            for t_idx, t_key in enumerate(true_keys):
                if t_idx not in matched_true_indices and p_key == t_key:
                    matched_true_indices.add(t_idx)
                    match_found = True
                    break

            if match_found:
                total_tp += 1
                per_type_tp[p_type] = per_type_tp.get(p_type, 0) + 1
            else:
                total_fp += 1
                per_type_fp[p_type] = per_type_fp.get(p_type, 0) + 1

        for t_idx, t_key in enumerate(true_keys):
            if t_idx not in matched_true_indices:
                t_type = true_list[t_idx].entity_type
                total_fn += 1
                per_type_fn[t_type] = per_type_fn.get(t_type, 0) + 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    per_type_metrics: dict[str, dict[str, float]] = {}

    for ent_type in sorted(all_types):
        tp = per_type_tp.get(ent_type, 0)
        fp = per_type_fp.get(ent_type, 0)
        fn = per_type_fn.get(ent_type, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_type_metrics[ent_type] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(score),
            "support": int(tp + fn),
        }

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "total_true_entities": total_tp + total_fn,
        "total_pred_entities": total_tp + total_fp,
        "per_type": per_type_metrics,
    }


class InferenceProfiler:
    """Measures model inference latency and throughput."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = str(device)
        self._start_time: float = 0.0
        self.elapsed_time: float = 0.0
        self.num_documents: int = 0

    def start(self) -> None:
        """Begin profiling session."""
        self._start_time = time.perf_counter()

    def stop(self, num_documents: int = 1) -> dict[str, float]:
        """Stop profiling session and return performance dictionary."""
        self.elapsed_time = time.perf_counter() - self._start_time
        self.num_documents = max(num_documents, 1)

        latency = (self.elapsed_time / self.num_documents) * 1000.0
        throughput = self.num_documents / self.elapsed_time if self.elapsed_time > 0 else 0.0

        return {
            "elapsed_sec": float(self.elapsed_time),
            "latency": float(latency),
            "throughput": float(throughput),
        }
