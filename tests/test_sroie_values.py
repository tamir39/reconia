"""Original SROIE keys must be scored independently of OCR alignment."""

import json
import pytest

from app.evaluation.sroie_values import compute_sroie_metrics, score_files


def test_multiline_ocr_and_complete_vlm_value_receive_same_score():
    targets = {"r": [{"fieldtype": "address", "text": "12 MAIN ST FLOOR 2"},
                     {"fieldtype": "total", "text": "10.00"}]}
    ocr = {"r": [{"fieldtype": "address", "text": "12 MAIN ST"},
                 {"fieldtype": "address", "text": "FLOOR 2"},
                 {"fieldtype": "total", "text": "10.00"}]}
    vlm = {"r": [{"fieldtype": "address", "text": "１２ Main St\nFloor 2"},
                 {"fieldtype": "total", "text": "10.00"}]}
    assert compute_sroie_metrics(ocr, targets)[0]["f1"] == 1.0
    assert compute_sroie_metrics(vlm, targets)[0]["f1"] == 1.0
    # Extra or repeated fragments cannot be silently selected away using labels.
    ocr["r"].append({"fieldtype": "address", "text": "12 MAIN ST"})
    assert compute_sroie_metrics(ocr, targets)[0]["f1"] == 0.5


def test_original_value_not_read_by_ocr_remains_a_false_negative():
    targets = {"r": [{"fieldtype": "company", "text": "UNREADABLE COMPANY"}]}
    result, _ = compute_sroie_metrics({"r": []}, targets)
    assert result["support"] == 1 and result["recall"] == 0


def test_missing_annotation_is_masked_but_prediction_is_retained():
    pred = {"r": [{"fieldtype": "total", "text": "10.00"},
                  {"fieldtype": "address", "text": "NOT ANNOTATED"}]}
    result, aggregated = compute_sroie_metrics(pred, {"r": [pred["r"][0]]})
    assert result["f1"] == 1 and result["unannotated_field_slots"] == 3
    assert len(aggregated["r"]) == 2


def test_four_field_score_rejects_mismatched_documents_and_unknown_keys():
    with pytest.raises(ValueError, match="IDs"):
        compute_sroie_metrics({}, {"r": []})
    with pytest.raises(ValueError, match="Unknown"):
        compute_sroie_metrics({"r": [{"fieldtype": "other", "text": "x"}]}, {"r": []})


def test_file_scoring_requires_full_split_unless_explicitly_limited(tmp_path):
    pred = tmp_path / "pred.json"
    target = tmp_path / "target.json"
    out = tmp_path / "score.json"
    pred.write_text(json.dumps({"r1": [{"fieldtype": "total", "text": "10"}]}))
    target.write_text(json.dumps({"val": {"r1": [{"fieldtype": "total", "text": "10"}],
                                         "r2": [{"fieldtype": "total", "text": "20"}]}}))
    with pytest.raises(ValueError, match="IDs"):
        score_files(pred, target, "val", out)
    result = score_files(pred, target, "val", out, limit=1)
    assert not result["complete"] and result["expected_num_documents"] == 2
    assert out.with_suffix(".values.json").is_file()
