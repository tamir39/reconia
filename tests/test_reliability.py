"""Regression checks for reproducibility and benchmark evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from app.data.schema import BBox, Document, Field, Word
from app.evaluation.metrics import EntitySpan, compute_entity_metrics, compute_token_metrics
from app.evaluation.predict import _resolve_checkpoint, prediction_filename, run_prediction_pipeline
from app.evaluation.report import export_full_benchmark_report
from app.evaluation.values import compute_value_metrics
from app.models_dl.gnn import InvoiceGNN, InvoiceGNNConfig, TEXT_HASH_VERSION, validate_gnn_checkpoint


def test_checkpoint_encoding_survives_a_new_python_process(tmp_path):
    code = '''
import json, sys, torch
from pathlib import Path
from app.models_dl.gnn import SubwordTextEncoder
p = Path(sys.argv[1])
torch.manual_seed(42)
m = SubwordTextEncoder(vocab_size=300, embedding_dim=4)
if p.exists():
    m.load_state_dict(torch.load(p, weights_only=True))
else:
    torch.save(m.state_dict(), p)
print(json.dumps(m(["Invoice", "Đơn hàng", ""], torch.device("cpu")).detach().tolist()))
'''
    outputs = []
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "encoder.pt")],
            env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True,
            check=True, timeout=120,
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]


def test_legacy_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="Retrain"):
        validate_gnn_checkpoint({"model_state_dict": {}})
    validate_gnn_checkpoint({"text_hash_version": TEXT_HASH_VERSION})


def test_false_prediction_of_an_absent_class_counts():
    metrics = compute_token_metrics([1, 0], [1, 2], ["O", "B-total", "B-date"])
    assert metrics["precision"] == 0.5
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_entity_batches_cannot_silently_drop_documents():
    with pytest.raises(ValueError, match="same number"):
        compute_entity_metrics([[], []], [[]])


def test_wrong_architecture_is_never_selected(tmp_path):
    (tmp_path / "roberta_docile_best.pt").touch()
    assert _resolve_checkpoint("gnn", "docile", tmp_path) is None


def test_sample_export_has_a_different_name():
    assert prediction_filename("gnn", "docile", "test", 2) != prediction_filename("gnn", "docile", "test", None)


def test_prediction_counts_blank_docs_and_unseen_types_and_keeps_pages(tmp_path, monkeypatch):
    import app.evaluation.predict as predict

    box = BBox(0, 0, 0.2, 0.1)
    docs = {
        "positive": Document("positive", [Word("100", box, 1)], [Field("total", box, 1, "100")], 2),
        "negative": Document("negative", [Word("noise", box, 0)], []),
        "unseen": Document("unseen", [Word("rare", box, 0)], [Field("rare", box, 0, "rare")]),
    }

    class Adapter:
        def splits(self): return ["test"]
        def document_ids(self, split): return list(docs)
        def load(self, doc_id): return docs[doc_id]

    def fake_predict(model, doc, vocab, inv_vocab, device):
        if doc.doc_id == "unseen": return [], []
        page = doc.words[0].page
        text = doc.words[0].text
        return [EntitySpan("total", 0, 1, text, page)], [{"fieldtype": "total", "text": text, "page": page}]

    monkeypatch.setitem(predict.ADAPTERS, "docile", Adapter)
    monkeypatch.setattr(predict, "predict_document_gnn", fake_predict)
    cfg = InvoiceGNNConfig(text_dim=4, hidden_dim=4, edge_dim=4, num_layers=1, num_classes=3)
    checkpoint = tmp_path / "gnn.pt"
    torch.save({"config": cfg, "model_state_dict": InvoiceGNN(cfg).state_dict(),
                "vocabulary": {"O": 0, "B-total": 1, "I-total": 2},
                "text_hash_version": TEXT_HASH_VERSION}, checkpoint)
    output = tmp_path / "predictions.json"
    result = run_prediction_pipeline(checkpoint, "docile", "gnn", output_json=output, device_str="cpu")
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["entity_metrics"]["per_type"]["rare"]["support"] == 1
    assert result["complete"] is True
    assert len(json.loads(output.read_text())) == 3
    evidence = json.loads(output.with_suffix(".metrics.json").read_text())
    assert len(evidence["checkpoint_sha256"]) == len(evidence["prediction_sha256"]) == 64
    sample = run_prediction_pipeline(checkpoint, "docile", "gnn", limit=1, device_str="cpu")
    assert sample["complete"] is False
    assert sample["expected_num_documents"] == 3


@pytest.mark.parametrize("requested,actual", [("val", "validation"), ("validation", "val")])
def test_prediction_resolves_validation_alias(tmp_path, monkeypatch, requested, actual):
    import app.evaluation.predict as predict

    class Adapter:
        def splits(self): return ["train", actual, "test"]
        def document_ids(self, split):
            assert split == actual
            return ["receipt"]
        def has_ground_truth(self, split):
            assert split == actual
            return True
        def load(self, doc_id): return Document(doc_id, [], [])

    monkeypatch.setattr(predict, "load_adapter", lambda *args: Adapter())
    cfg = InvoiceGNNConfig(text_dim=4, hidden_dim=4, edge_dim=4, num_layers=1, num_classes=3)
    checkpoint = tmp_path / "gnn.pt"
    torch.save({"config": cfg, "model_state_dict": InvoiceGNN(cfg).state_dict(),
                "vocabulary": {"O": 0, "B-total": 1, "I-total": 2},
                "text_hash_version": TEXT_HASH_VERSION}, checkpoint)
    result = run_prediction_pipeline(checkpoint, "cord", "gnn", split=requested, device_str="cpu")
    assert result["split"] == requested
    assert result["source_split"] == actual
    assert result["complete"] and result["num_documents"] == 1
    with pytest.raises(ValueError, match="not found"):
        run_prediction_pipeline(checkpoint, "cord", "gnn", split="unknown", device_str="cpu")


def test_report_uses_test_per_type_instead_of_validation(tmp_path, monkeypatch):
    import app.evaluation.report as report
    monkeypatch.setattr(report, "plt", None)
    result = export_full_benchmark_report(
        output_dir=tmp_path, model_name="GNN", dataset_name="fake",
        history=[{"epoch": 1, "train_loss": 0.3, "val_loss": 0.2}], metadata={"best_epoch": 1},
        final_val_metrics={"per_type": {"validation_only": {"f1": 1.0}}},
        test_metrics={"complete": True, "per_type": {"test_only": {"f1": 0.5}}},
    )
    saved = json.loads(result["json"].read_text())
    assert set(saved["per_type"]) == {"test_only"}
    with pytest.raises(ValueError, match="partial"):
        export_full_benchmark_report(
            tmp_path, "GNN", "fake", [], {}, {}, test_metrics={"complete": False},
        )


def test_value_metric_keeps_duplicates_and_fields_without_ocr():
    pred = {"doc": [{"fieldtype": "total", "text": " ＡＢＣ  "}]}
    targets = {"doc": [{"fieldtype": "total", "text": "abc"},
                       {"fieldtype": "total", "text": "abc"},
                       {"fieldtype": "date", "text": "2026-09-24"}]}
    result = compute_value_metrics(pred, targets)
    assert result["precision"] == 1.0
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == 0.5
    with pytest.raises(ValueError, match="IDs"):
        compute_value_metrics({}, targets)
