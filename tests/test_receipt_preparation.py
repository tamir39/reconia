"""Upload preparation must reject unexpected files and changed sources."""

import json
import hashlib

import pytest

from scripts.prepare_receipts import prepare_sroie, receipt_files
from scripts.fetch_kaggle_receipts import receipt_digest, verify_cord


def make_cord(root):
    root.mkdir()
    for name in ("dataset_dict.json", "field-value-targets.json"):
        (root / name).write_text("{}")
    for split in ("train", "validation", "test"):
        folder = root / split
        folder.mkdir()
        (folder / "dataset_info.json").write_text("{}")
        (folder / "state.json").write_text(json.dumps({
            "_data_files": [{"filename": "data-00000-of-00001.arrow"}]}))
        (folder / "data-00000-of-00001.arrow").write_bytes(b"placeholder")


def test_bundle_rejects_an_unexpected_secret_file(tmp_path):
    make_cord(tmp_path / "cord")
    assert len(receipt_files(tmp_path, "cord")) == 11
    (tmp_path / "cord/.env").write_text("EXAMPLE=do-not-upload")
    with pytest.raises(ValueError, match="Unreviewed or missing"):
        receipt_files(tmp_path, "cord")


def test_bundle_rejects_escaping_shard_paths(tmp_path):
    make_cord(tmp_path / "cord")
    (tmp_path / "cord/train/state.json").write_text(json.dumps({
        "_data_files": [{"filename": "../../outside.arrow"}]}))
    with pytest.raises(ValueError, match="Unexpected CORD shard"):
        receipt_files(tmp_path, "cord")


def test_changed_sroie_release_is_not_silently_accepted(tmp_path):
    (tmp_path / "SROIE2019.zip").write_bytes(b"unexpected release bytes")
    with pytest.raises(ValueError, match="differs from the reviewed archive"):
        prepare_sroie(tmp_path)


def test_source_mapping_is_portable_but_content_changes_are_rejected(tmp_path):
    path = tmp_path / "source-id-map.json"
    path.write_bytes(b'{\r\n  "source_ids": {"000": "X1"}\r\n}\r\n')
    expected = receipt_digest(path)
    path.write_bytes(b'{\n  "source_ids": {"000": "X1"}\n}\n')
    assert receipt_digest(path) == expected
    path.write_bytes(b'{"source_ids":{"000":"X2"}}')
    assert receipt_digest(path) != expected


@pytest.mark.parametrize("mutation", ["image", "annotation", "order", "missing_split"])
def test_cord_download_must_match_reviewed_content(mutation):
    from datasets import Dataset, Features, Image, Value

    images = [b"receipt-one", b"receipt-two"]
    annotations = ['{"total":"10"}', '{"total":"20"}']
    expected = {"test": [
        {"image_sha256": hashlib.sha256(image).hexdigest(),
         "ground_truth_sha256": hashlib.sha256(
             json.dumps(json.loads(gt), sort_keys=True, ensure_ascii=False).encode()
         ).hexdigest()}
        for image, gt in zip(images, annotations)
    ]}

    def dataset():
        return {"test": Dataset.from_dict(
            {"image": images, "ground_truth": annotations},
            features=Features({"image": Image(decode=False), "ground_truth": Value("string")}),
        )}

    verify_cord(dataset(), expected)
    if mutation == "image":
        images[0] = b"different-receipt"
    elif mutation == "annotation":
        annotations[0] = '{"total":"999"}'
    elif mutation == "order":
        images.reverse()
        annotations.reverse()
    elif mutation == "missing_split":
        expected["train"] = []
    with pytest.raises(ValueError, match="differ"):
        verify_cord(dataset(), expected)
