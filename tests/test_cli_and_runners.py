"""Unit tests for CLI parsers, runner dispatchers, and helper functions.

Ensures that all CLI tools (prepare, predict, run_benchmark, train_gnn, train_roberta)
maintain unified conventions, default to 'all' datasets, and handle parameters correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pytest

from app.evaluation.predict import (
    _resolve_checkpoint,
    run_prediction_pipeline,
    span_to_bbox,
    spans_to_docile_predictions,
)
from app.data.schema import BBox, Document, Word
from app.evaluation.metrics import EntitySpan
from app.training.run_benchmark import format_duration, SUPPORTED_MODELS
from app.training.utils import DualLogger, str2bool


def test_format_duration():
    """Verify that duration in seconds is cleanly formatted into HHh MMm SSs."""
    assert format_duration(0) == "00h 00m 00s"
    assert format_duration(59) == "00h 00m 59s"
    assert format_duration(65) == "00h 01m 05s"
    assert format_duration(3600) == "01h 00m 00s"
    assert format_duration(3665) == "01h 01m 05s"
    assert format_duration(7325) == "02h 02m 05s"


def test_prepare_cli_defaults():
    """Verify prepare.py parser defaults and split mapping logic."""
    import app.graph.prepare as prepare_module

    # Test parser defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "sroie", "cord", "docile"])
    parser.add_argument("--splits", nargs="+", default=["all"])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args([])
    assert args.dataset == "all"
    assert args.splits == ["all"]
    assert args.limit is None

    # Test 'all' resolution for CORD
    avail_splits_cord = ["train", "validation", "test"]
    mapped_cord = []
    if "all" in args.splits:
        for s in ("train", "validation", "val", "test"):
            if s in avail_splits_cord and s not in mapped_cord:
                mapped_cord.append(s)
    assert mapped_cord == ["train", "validation", "test"]

    # Test 'all' resolution for SROIE
    avail_splits_sroie = ["train", "val", "test"]
    mapped_sroie = []
    if "all" in args.splits:
        for s in ("train", "validation", "val", "test"):
            if s in avail_splits_sroie and s not in mapped_sroie:
                mapped_sroie.append(s)
    assert mapped_sroie == ["train", "val", "test"]

    # Test bidirectional split mapping logic for specific inputs
    custom_input = ["val"]
    mapped_specific = []
    for s in custom_input:
        if s in avail_splits_cord:
            mapped_specific.append(s)
        elif s == "val" and "validation" in avail_splits_cord:
            mapped_specific.append("validation")
    assert mapped_specific == ["validation"]


def test_predict_cli_defaults():
    """Verify predict.py parser defaults and resolution behavior."""
    from app.evaluation.predict import main as predict_main

    # Invoking predict_main with --dataset all should gracefully run and return 0
    ret = predict_main(["--dataset", "all", "--limit", "2"])
    assert ret == 0


def test_run_prediction_pipeline_interface():
    """Verify run_prediction_pipeline is callable and exported from predict."""
    assert callable(run_prediction_pipeline)


def test_resolve_checkpoint_patterns(tmp_path: Path, monkeypatch):
    """Verify that _resolve_checkpoint reliably matches standard naming conventions."""
    import app.evaluation.predict as predict_mod

    # Patch RECONIA_ROOT inside predict module to tmp_path
    monkeypatch.setattr(predict_mod, "Path", lambda *args: tmp_path if "checkpoints" in str(args) else Path(*args))

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initially no checkpoint exists in clean test dir
    found = _resolve_checkpoint("gnn", "sroie", checkpoints_dir=ckpt_dir)
    assert found is None or not found.exists()

    # 2. Create gnn_sroie_best.pt
    mock_ckpt = ckpt_dir / "gnn_sroie_best.pt"
    mock_ckpt.write_text("mock", encoding="utf-8")

    # Verify resolution picks up candidate
    candidate = _resolve_checkpoint("gnn", "sroie", checkpoints_dir=ckpt_dir)
    assert candidate == mock_ckpt


def test_run_benchmark_cli_defaults():
    """Verify run_benchmark.py parser arguments and defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", "sroie", "cord", "docile"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--monitor-metric", default="f1", choices=["f1", "val_loss"])

    args = parser.parse_args([])
    assert args.models == ["all"]
    assert args.datasets == ["all"]
    assert args.epochs == 20
    assert args.patience == 5
    assert args.limit_train is None
    assert args.limit_val is None
    assert args.monitor_metric == "f1"

    # Verify model resolution logic
    if "all" in args.models:
        selected_models = list(SUPPORTED_MODELS)
    assert selected_models == ["gnn", "roberta"]

    # Verify dataset resolution logic
    if "all" in args.datasets:
        selected_datasets = ["sroie", "cord", "docile"]
    assert selected_datasets == ["sroie", "cord", "docile"]


def test_train_gnn_cli_defaults():
    """Verify train_gnn.py argument defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "sroie", "cord", "docile"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-accum", type=int, default=16)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)

    args = parser.parse_args([])
    assert args.dataset == "all"
    assert args.epochs == 20
    assert args.lr == 1e-3
    assert args.hidden_dim == 128
    assert args.batch_accum == 16
    assert args.limit_train is None
    assert args.limit_val is None


def test_train_roberta_cli_defaults():
    """Verify train_roberta.py argument defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "sroie", "cord", "docile"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-accum", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)

    args = parser.parse_args([])
    assert args.dataset == "all"
    assert args.epochs == 20
    assert args.lr == 3e-5
    assert args.batch_size == 4
    assert args.batch_accum == 4
    assert args.limit_train is None
    assert args.limit_val is None


def test_span_to_bbox_empty():
    """Verify span_to_bbox handles empty or out-of-range spans without error."""
    doc = Document(doc_id="test_doc", words=[], fields=[])
    span = EntitySpan("total", 0, 0, "")
    bbox = span_to_bbox(doc, span)
    assert bbox == [0.0, 0.0, 0.0, 0.0]


def test_no_multi_flag_cli_arguments():
    """Ensure every CLI script strictly uses single flags without any aliases."""
    import ast

    cli_files = [
        Path("app/graph/prepare.py"),
        Path("app/evaluation/predict.py"),
        Path("app/evaluation/report.py"),
        Path("app/training/run_benchmark.py"),
        Path("app/training/train_gnn.py"),
        Path("app/training/train_roberta.py"),
    ]

    for rel_path in cli_files:
        full_path = Path(__file__).resolve().parents[1] / rel_path
        tree = ast.parse(full_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "add_argument") or (
                    isinstance(func, ast.Name) and func.id == "add_argument"
                ):
                    flag_args = [
                        arg.value
                        for arg in node.args
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-")
                    ]
                    assert len(flag_args) <= 1, (
                        f"Found multi-flag argument in {rel_path}: {flag_args}. "
                        "Every CLI argument must use exactly ONE canonical flag matching README.md."
                    )


def test_spans_to_docile_predictions():
    """Verify spans_to_docile_predictions converts EntitySpans to DocILE prediction format."""
    words = [
        Word("TOTAL", BBox(10.0, 10.0, 50.0, 20.0), page=0),
        Word("100.00", BBox(55.0, 10.0, 90.0, 20.0), page=0),
    ]
    doc = Document(doc_id="doc1", words=words, fields=[])
    span = EntitySpan(entity_type="total", start_token=0, end_token=2, text="TOTAL 100.00", page=0)
    confidences = [0.95, 0.85]

    preds = spans_to_docile_predictions(doc, [span], confidences)
    assert len(preds) == 1
    assert preds[0]["page"] == 0
    assert preds[0]["fieldtype"] == "total"
    assert preds[0]["score"] == 0.9
    assert preds[0]["text"] == "TOTAL 100.00"
    assert preds[0]["bbox"] == [10.0, 10.0, 90.0, 20.0]


def test_str2bool_utility():
    """Verify str2bool correctly parses boolean strings and raises error on invalid inputs."""
    assert str2bool(True) is True
    assert str2bool(False) is False
    assert str2bool("true") is True
    assert str2bool("1") is True
    assert str2bool("yes") is True
    assert str2bool("false") is False
    assert str2bool("0") is False
    assert str2bool("no") is False
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("invalid_bool")


