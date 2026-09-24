"""Upload preparation must reject unexpected files and changed sources."""

import json

import pytest

from scripts.prepare_receipts import prepare_sroie, receipt_files


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
