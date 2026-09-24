import pytest

from app.data.export import export_dataset
from app.data.portable import ExportAdapter, load_adapter
from app.data.schema import BBox, Document, Word


class TinyAdapter:
    name = "fixture"

    def splits(self):
        return ["train", "val", "test"]

    def iter_documents(self, split):
        yield Document(split, [Word("invoice", BBox(0, 0, 1, 1), 0)])


def test_portable_adapter_integrity_and_explicit_selection(tmp_path, monkeypatch):
    export_dataset(TinyAdapter(), tmp_path, splits=("train", "val", "test"), shards=1)
    monkeypatch.setenv("FIXTURE_EXPORT_DIR", str(tmp_path))
    adapter = load_adapter("fixture", lambda: pytest.fail("raw fallback used"))
    assert adapter.document_ids("validation") == ["val"]
    assert next(adapter.iter_documents("train")).words[0].text == "invoice"
    with pytest.raises(NotImplementedError):
        adapter.get_image("train")
    with pytest.raises(ValueError, match="match"):
        ExportAdapter(tmp_path, "docile")
    (tmp_path / "train-000.jsonl.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ExportAdapter(tmp_path, "fixture")


def test_portable_adapter_rejects_split_leakage(tmp_path):
    class Overlapping(TinyAdapter):
        def iter_documents(self, split):
            yield Document("same-id")
    export_dataset(Overlapping(), tmp_path, splits=("train", "val"), shards=1)
    with pytest.raises(ValueError, match="overlapping"):
        ExportAdapter(tmp_path, "fixture")
