"""Read verified OCR exports on Kaggle without raw dataset dependencies."""

import json
import os
from pathlib import Path

from app.data.export import document_from_dict, load_manifest, verify
import gzip


class ExportAdapter:
    """In-memory OCR-only adapter; images must be provided separately for VLMs."""

    def __init__(self, root, dataset):
        self.root = Path(root)
        manifest = load_manifest(self.root)
        if manifest["dataset"] != dataset:
            raise ValueError("Export dataset does not match requested dataset")
        if verify(self.root):
            raise ValueError("Export checksum verification failed")
        self.name = dataset
        self._documents = {}
        self._splits = {}
        for split, info in manifest["splits"].items():
            ids = []
            for shard in info["shards"]:
                path = (self.root / shard).resolve()
                if not path.is_relative_to(self.root.resolve()):
                    raise ValueError("Shard path escapes export directory")
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        doc = document_from_dict(json.loads(line))
                        if doc.doc_id in self._documents:
                            raise ValueError("Duplicate document ID or overlapping splits")
                        self._documents[doc.doc_id] = doc
                        ids.append(doc.doc_id)
            if len(ids) != info["documents"]:
                raise ValueError("Export document count differs from manifest")
            self._splits[split] = sorted(ids)
        # Keep the exact order and membership of the versioned local split.
        split_file = Path(__file__).with_name(f"{dataset}_splits.json")
        if split_file.exists():
            expected = json.loads(split_file.read_text(encoding="utf-8"))
            for split in ("train", "val", "test"):
                if set(self._splits.get(split, [])) != set(expected[split]):
                    raise ValueError(f"Export {split} IDs differ from versioned split; re-export data")
                self._splits[split] = expected[split]

    def splits(self):
        return list(self._splits)

    def document_ids(self, split):
        return list(self._splits["val" if split == "validation" else split])

    def load(self, doc_id):
        return self._documents[doc_id]

    def iter_documents(self, split, limit=None):
        ids = self.document_ids(split)
        for doc_id in ids if limit is None else ids[:limit]:
            yield self.load(doc_id)

    def has_ground_truth(self, split):
        return split in self._splits

    def get_image(self, doc_id, page=0):
        image_root = os.environ.get(f"{self.name.upper()}_IMAGE_DIR")
        if image_root:
            if not hasattr(self, "_image_store"):
                from app.data.page_images import PageImageStore
                self._image_store = PageImageStore(image_root, self)
            return self._image_store.get_image(doc_id, page)
        raise NotImplementedError("OCR exports contain no page images; attach the original dataset for image models")


def load_adapter(dataset, fallback):
    """Explicit export setting takes precedence over raw dataset discovery."""
    root = os.environ.get(f"{dataset.upper()}_EXPORT_DIR")
    return ExportAdapter(root, dataset) if root else fallback()
