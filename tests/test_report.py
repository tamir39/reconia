"""Tests for training evaluation and report generation."""

import csv
import json
from pathlib import Path

import pytest

from app.evaluation.report import (
    export_full_benchmark_report,
    generate_benchmark_summary,
    plot_training_dashboard,
    save_markdown_report,
    save_training_history_csv,
)

HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "precision",
    "recall",
    "f1",
    "latency_ms",
    "peak_vram_mb",
]

SAMPLE_HISTORY = [
    {
        "epoch": 1,
        "train_loss": 0.5,
        "val_loss": 0.4,
        "precision": 0.6,
        "recall": 0.5,
        "f1": 0.55,
        "latency_ms": 5.0,
        "peak_vram_mb": 64.0,
    },
    {
        "epoch": 2,
        "train_loss": 0.2,
        "val_loss": 0.15,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "latency_ms": 4.8,
        "peak_vram_mb": 64.0,
    },
]


def test_save_training_history_csv(tmp_path: Path):
    csv_file = tmp_path / "history.csv"
    save_training_history_csv(csv_file, SAMPLE_HISTORY)

    assert csv_file.exists()
    with csv_file.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == HISTORY_COLUMNS
        rows = list(reader)
        assert len(rows) == 2
        assert float(rows[0]["precision"]) == pytest.approx(0.6)
        assert float(rows[1]["f1"]) == pytest.approx(0.75)


def test_save_markdown_report(tmp_path: Path):
    md_file = tmp_path / "report.md"
    summary_data = {
        "model": "GNN",
        "dataset": "sroie",
        "device": "cuda",
        "total_params": 50000,
        "best_epoch": 2,
        "train_loss": 0.2,
        "val_loss": 0.15,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "latency_ms": 4.8,
        "peak_vram_mb": 64.0,
        "history": SAMPLE_HISTORY,
        "entity_per_type": {
            "company": {"precision": 0.85, "recall": 0.8, "f1": 0.82, "support": 50},
            "total": {"precision": 0.75, "recall": 0.7, "f1": 0.72, "support": 30},
        },
    }

    save_markdown_report(md_file, summary_data)
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")

    assert "Precision" in content
    assert "Recall" in content
    assert "F1" in content
    assert "Latency (ms)" in content
    assert "VRAM (MB)" in content
    assert "75.00%" in content
    assert "company" in content


def test_plot_training_dashboard(tmp_path: Path):
    fig_file = tmp_path / "curves.png"
    per_type = {
        "total": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 30},
        "date": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 30},
    }

    result = plot_training_dashboard(fig_file, SAMPLE_HISTORY, per_type, "GNN", "SROIE")
    if result is not None:
        assert fig_file.exists()
        assert fig_file.stat().st_size > 1000


def test_export_full_benchmark_report(tmp_path: Path):
    metadata = {
        "device": "cuda",
        "total_params": 50000,
        "best_epoch": 2,
    }
    final_metrics = {
        "entity_metrics": {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "per_type": {
                "total": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 30}
            },
        }
    }

    out = export_full_benchmark_report(
        output_dir=tmp_path,
        model_name="GNN",
        dataset_name="SROIE",
        history=SAMPLE_HISTORY,
        final_val_metrics=final_metrics,
        metadata=metadata,
    )

    assert out["json"].exists()
    assert out["csv"].exists()
    assert out["markdown"].exists()

    with out["json"].open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["f1"] == pytest.approx(0.75)
    assert data["precision"] == pytest.approx(0.8)
    assert data["recall"] == pytest.approx(0.7)


def test_generate_benchmark_summary(tmp_path: Path):
    record_gnn = {
        "model": "GNN",
        "dataset": "sroie",
        "best_epoch": 2,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "latency_ms": 4.8,
        "peak_vram_mb": 64.0,
    }
    record_roberta = {
        "model": "RoBERTa",
        "dataset": "sroie",
        "best_epoch": 1,
        "precision": 0.6,
        "recall": 0.5,
        "f1": 0.55,
        "latency_ms": 50.0,
        "peak_vram_mb": 1500.0,
    }

    with (tmp_path / "gnn_sroie_results.json").open("w", encoding="utf-8") as fh:
        json.dump(record_gnn, fh)
    with (tmp_path / "roberta_sroie_results.json").open("w", encoding="utf-8") as fh:
        json.dump(record_roberta, fh)

    summary = generate_benchmark_summary(tmp_path)
    assert summary["markdown"] is not None
    assert summary["markdown"].exists()

    md_content = summary["markdown"].read_text(encoding="utf-8")
    assert "Precision (%)" in md_content
    assert "Recall (%)" in md_content
    assert "F1 (%)" in md_content
    assert "Latency (ms)" in md_content
    assert "VRAM (MB)" in md_content
    assert "GNN" in md_content
    assert "RoBERTa" in md_content
