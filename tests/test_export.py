"""Export round-trip and integrity."""

import gzip
import json

import pytest

from app.data.export import (
    document_from_dict,
    document_to_dict,
    export_dataset,
    load_manifest,
    read_shards,
    verify,
    write_shards,
)
from app.data.schema import BBox, Document, Field, Word


def make_doc(doc_id, n_words=3):
    words = [
        Word(text=f"w{i}", bbox=BBox(0.1 * i, 0.1, 0.1 * i + 0.05, 0.13), page=0, confidence=0.9)
        for i in range(n_words)
    ]
    fields = [
        Field("vendor_name", BBox(0.0, 0.09, 0.3, 0.14), 0, "w0 w1"),
        Field("line_item_quantity", BBox(0.0, 0.2, 0.1, 0.24), 0, "5", line_item_id=1),
    ]
    return Document(doc_id=doc_id, words=words, fields=fields, page_count=1, source="test")


class FakeAdapter:
    name = "fake"

    def __init__(self, counts):
        self.counts = counts

    def splits(self):
        return list(self.counts)

    def document_ids(self, split):
        return [f"{split}-{i}" for i in range(self.counts[split])]

    def load(self, doc_id):
        return make_doc(doc_id)

    def iter_documents(self, split, limit=None):
        ids = self.document_ids(split)
        for doc_id in ids[:limit] if limit else ids:
            yield self.load(doc_id)


def test_document_survives_dict_round_trip():
    original = make_doc("a")
    restored = document_from_dict(document_to_dict(original))

    assert restored.doc_id == original.doc_id
    assert restored.page_count == original.page_count
    assert [w.text for w in restored.words] == [w.text for w in original.words]
    assert [f.fieldtype for f in restored.fields] == [f.fieldtype for f in original.fields]


def test_line_item_id_is_preserved():
    """Dropping it would silently destroy the LIR labels."""
    restored = document_from_dict(document_to_dict(make_doc("a")))
    line_items = [f for f in restored.fields if f.is_line_item]
    assert len(line_items) == 1
    assert line_items[0].line_item_id == 1


def test_absent_confidence_stays_absent():
    doc = Document(
        doc_id="x",
        words=[Word("w", BBox(0.1, 0.1, 0.2, 0.2), 0, confidence=None)],
    )
    payload = document_to_dict(doc)
    assert "c" not in payload["words"][0]
    assert document_from_dict(payload).words[0].confidence is None


def test_coordinates_keep_enough_precision():
    doc = Document(doc_id="x", words=[Word("w", BBox(0.123456789, 0.2, 0.3, 0.4), 0)])
    box = document_from_dict(document_to_dict(doc)).words[0].bbox
    assert abs(box.x0 - 0.123456789) < 1e-5


def test_shards_round_trip(tmp_path):
    docs = [make_doc(f"d{i}") for i in range(10)]
    counts = write_shards(docs, tmp_path, "train", shards=3)

    assert counts.documents == 10
    assert counts.words == 30
    assert counts.fields == 20

    recovered = list(read_shards(tmp_path, "train"))
    assert len(recovered) == 10
    assert {d.doc_id for d in recovered} == {f"d{i}" for i in range(10)}


def test_documents_are_spread_across_shards(tmp_path):
    """Round-robin, so a partially downloaded export is still representative."""
    write_shards([make_doc(f"d{i}") for i in range(9)], tmp_path, "train", shards=3)
    for i in range(3):
        with gzip.open(tmp_path / f"train-{i:03d}.jsonl.gz", "rt", encoding="utf-8") as fh:
            assert len([ln for ln in fh if ln.strip()]) == 3


def test_more_shards_than_documents_is_safe(tmp_path):
    write_shards([make_doc("only")], tmp_path, "train", shards=5)
    assert len(list(read_shards(tmp_path, "train"))) == 1


def test_splits_do_not_bleed_into_each_other(tmp_path):
    export_dataset(FakeAdapter({"train": 6, "val": 2}), tmp_path, shards=2)
    train = {d.doc_id for d in read_shards(tmp_path, "train")}
    val = {d.doc_id for d in read_shards(tmp_path, "val")}
    assert train & val == set()
    assert len(train) == 6 and len(val) == 2


def test_manifest_records_counts_and_fieldtypes(tmp_path):
    manifest = export_dataset(FakeAdapter({"train": 4, "val": 1}), tmp_path, shards=2)
    assert manifest["dataset"] == "fake"
    assert manifest["splits"]["train"]["documents"] == 4
    assert manifest["splits"]["val"]["documents"] == 1
    assert set(manifest["fieldtypes"]) == {"vendor_name", "line_item_quantity"}
    assert load_manifest(tmp_path)["version"] == manifest["version"]


def test_verify_passes_on_an_intact_export(tmp_path):
    export_dataset(FakeAdapter({"train": 4}), tmp_path, splits=("train",), shards=2)
    assert verify(tmp_path) == []


def test_verify_detects_a_corrupted_shard(tmp_path):
    """A truncated upload must surface as an error, not as unexplained accuracy loss."""
    export_dataset(FakeAdapter({"train": 4}), tmp_path, splits=("train",), shards=2)
    target = tmp_path / "train-000.jsonl.gz"
    with gzip.open(target, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(document_to_dict(make_doc("tampered"))) + "\n")
    assert "train-000.jsonl.gz" in verify(tmp_path)


def test_unknown_split_raises_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="no split"):
        export_dataset(FakeAdapter({"train": 2}), tmp_path, splits=("test",), shards=1)


def test_manifest_rejects_a_foreign_version(tmp_path):
    export_dataset(FakeAdapter({"train": 2}), tmp_path, splits=("train",), shards=1)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(tmp_path)


def test_validation_alias_is_exported_as_canonical_val(tmp_path):
    manifest = export_dataset(FakeAdapter({"train": 2, "validation": 1}), tmp_path, shards=1)
    assert manifest["splits"]["val"]["documents"] == 1
    assert [doc.doc_id for doc in read_shards(tmp_path, "val")] == ["validation-0"]
