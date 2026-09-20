"""Unit tests for the unified evaluation metrics module."""

import numpy as np
import pytest

from app.evaluation.metrics import (
    EntitySpan,
    InferenceProfiler,
    compute_entity_metrics,
    compute_token_metrics,
    extract_entities_from_bio,
)


def test_extract_entities_from_bio():
    tags = ["O", "B-company", "I-company", "O", "B-total", "O", "B-date"]
    texts = ["Receipt", "ABC", "CORP", "Total:", "$100.00", "Date:", "2023-05-01"]
    pages = [0, 0, 0, 0, 0, 0, 0]

    spans = extract_entities_from_bio(tags, texts, pages)
    assert len(spans) == 3

    assert spans[0].entity_type == "company"
    assert spans[0].start_token == 1
    assert spans[0].end_token == 3
    assert spans[0].text == "ABC CORP"

    assert spans[1].entity_type == "total"
    assert spans[1].start_token == 4
    assert spans[1].end_token == 5
    assert spans[1].text == "$100.00"

    assert spans[2].entity_type == "date"
    assert spans[2].start_token == 6
    assert spans[2].end_token == 7
    assert spans[2].text == "2023-05-01"


def test_compute_token_metrics_perfect():
    y_true = [0, 1, 2, 0, 3, 0]
    y_pred = [0, 1, 2, 0, 3, 0]
    label_names = {0: "O", 1: "B-company", 2: "I-company", 3: "B-total"}

    metrics = compute_token_metrics(y_true, y_pred, label_names, ignore_labels={"O"})
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["num_tokens"] == 6


def test_compute_token_metrics_with_errors():
    # 6 mock tokens representing words in an invoice:
    # [0: 'O', 1: 'B-company', 2: 'I-company', 3: 'O', 4: 'B-total', 5: 'O']
    y_true = [0, 1, 2, 0, 3, 0]
    # Prediction has 2 errors:
    # - index 2: true is 'I-company' (2), predicted as 'O' (0) -> False Negative for I-company
    # - index 5: true is 'O' (0), predicted as 'B-total' (3) -> False Positive for B-total
    y_pred = [0, 1, 0, 0, 3, 3]
    label_names = {0: "O", 1: "B-company", 2: "I-company", 3: "B-total"}

    metrics = compute_token_metrics(y_true, y_pred, label_names, ignore_labels={"O"})

    # Accuracy counts all tokens (4 correct out of 6):
    assert metrics["accuracy"] == 4 / 6
    assert metrics["num_tokens"] == 6

    # Overall token metrics (excluding 'O'):
    # TP=2, FP=1, FN=1 -> Precision = 2/3, Recall = 2/3, F1 = 2/3
    assert pytest.approx(metrics["precision"], 1e-4) == 2 / 3
    assert pytest.approx(metrics["recall"], 1e-4) == 2 / 3
    assert pytest.approx(metrics["f1"], 1e-4) == 2 / 3

    # Per-class breakdown:
    # B-company: 1 TP, 0 FP, 0 FN -> perfect
    assert metrics["per_class"]["B-company"]["precision"] == 1.0
    assert metrics["per_class"]["B-company"]["recall"] == 1.0
    assert metrics["per_class"]["B-company"]["f1"] == 1.0

    # I-company: 0 TP, 0 FP, 1 FN -> missed
    assert metrics["per_class"]["I-company"]["precision"] == 0.0
    assert metrics["per_class"]["I-company"]["recall"] == 0.0
    assert metrics["per_class"]["I-company"]["f1"] == 0.0

    # B-total: 1 TP, 1 FP, 0 FN
    assert metrics["per_class"]["B-total"]["precision"] == 0.5
    assert metrics["per_class"]["B-total"]["recall"] == 1.0
    assert pytest.approx(metrics["per_class"]["B-total"]["f1"], 1e-4) == 2 / 3


def test_compute_entity_metrics():
    # Document 1: 1 match, 1 missing, 1 extra
    true_doc1 = [
        EntitySpan("company", 0, 2, "ABC CORP"),
        EntitySpan("total", 5, 6, "100.00"),
    ]
    pred_doc1 = [
        EntitySpan("company", 0, 2, "ABC CORP"),  # Match (TP)
        EntitySpan("date", 3, 4, "2023"),        # Extra (FP)
    ]
    # Document 2: 1 match
    true_doc2 = [EntitySpan("address", 1, 3, "Main St")]
    pred_doc2 = [EntitySpan("address", 1, 3, "Main St")]

    metrics = compute_entity_metrics([pred_doc1, pred_doc2], [true_doc1, true_doc2], match_mode="exact")

    # True entities: 3 (company, total, address). Pred: 2 (company, date, address).
    # TP = 2 (company, address), FP = 1 (date), FN = 1 (total).
    assert metrics["total_true_entities"] == 3
    assert metrics["total_pred_entities"] == 3
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert abs(metrics["f1"] - (2 / 3)) < 1e-5
    assert metrics["per_type"]["company"]["f1"] == 1.0
    assert metrics["per_type"]["address"]["f1"] == 1.0
    assert metrics["per_type"]["total"]["f1"] == 0.0


def test_inference_profiler():
    profiler = InferenceProfiler(device="cpu")
    profiler.start()
    # Simulate work
    import time
    time.sleep(0.01)
    stats = profiler.stop(num_documents=2)

    assert stats["elapsed_sec"] > 0
    assert stats["latency_ms_per_doc"] > 0
    assert stats["throughput_docs_per_sec"] > 0
    assert stats["peak_vram_mb"] == 0.0  # CPU has 0 VRAM
