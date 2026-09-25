"""Verified complete PDF-page images, bound to the portable OCR documents."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image

from app.data.export import document_to_dict

IMAGE_POLICY = "poppler-full-page-rgb-png-longest1600-v1"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document_sha256(document):
    payload = json.dumps(document_to_dict(document), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def render_pdf(pdf_path, destination, *, page=None, executable=None):
    """Render complete pages with Poppler; never extract an embedded image."""
    command = executable or os.environ.get("PDFTOPPM") or shutil.which("pdftoppm")
    if not command:
        raise RuntimeError("Install Poppler or set PDFTOPPM to pdftoppm")
    if page is not None and page < 0:
        raise ValueError("Page index must be nonnegative")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Render destination must be empty")
    args = [str(command), "-r", "200", "-scale-to", "1600", "-png"]
    if page is not None:
        args += ["-f", str(page + 1), "-l", str(page + 1)]
    subprocess.run(args + [str(pdf_path), str(destination / "page")],
                   check=True, capture_output=True, timeout=180)
    images = sorted(destination.glob("page-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    if not images:
        raise ValueError("PDF rendering produced no pages")
    return images


def render_page(pdf_path, page):
    from pypdf import PdfReader
    count = len(PdfReader(pdf_path).pages)
    if not 0 <= page < count:
        raise ValueError(f"Page {page} outside PDF with {count} pages")
    with tempfile.TemporaryDirectory(prefix="reconia-page-") as directory:
        path = render_pdf(pdf_path, directory, page=page)[0]
        with Image.open(path) as image:
            return image.convert("RGB")


class PageImageStore:
    """Fail closed on wrong OCR, splits, missing pages or changed image bytes."""

    def __init__(self, root, adapter):
        self.root = Path(root).resolve()
        manifest_path = self.root / "image-manifest.json"
        self.manifest_sha256 = sha256(manifest_path)
        self.manifest = json.loads(manifest_path.read_bytes())
        if self.manifest.get("policy") != IMAGE_POLICY or self.manifest.get("dataset") != adapter.name:
            raise ValueError("Image bundle dataset or policy mismatch")
        expected = {s: adapter.document_ids(s) for s in ("train", "val", "test")}
        if self.manifest.get("splits") != expected:
            raise ValueError("Image bundle split mismatch")
        ids = [i for values in expected.values() for i in values]
        if len(ids) != len(set(ids)) or set(self.manifest["documents"]) != set(ids):
            raise ValueError("Image bundle document membership mismatch")
        for doc_id in ids:
            doc = adapter.load(doc_id)
            item = self.manifest["documents"][doc_id]
            if item["document_sha256"] != document_sha256(doc) or len(item["pages"]) != doc.page_count:
                raise ValueError(f"Image/OCR alignment mismatch: {doc_id}")
            for page in item["pages"]:
                path = self._path(page["file"])
                if sha256(path) != page["sha256"]:
                    raise ValueError(f"Page image checksum mismatch: {path.name}")

    def _path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Page image path escapes bundle")
        return path

    def get_image(self, doc_id, page=0):
        pages = self.manifest["documents"][doc_id]["pages"]
        if not 0 <= page < len(pages):
            raise ValueError("Page index outside image bundle")
        info = pages[page]
        with Image.open(self._path(info["file"])) as image:
            if list(image.size) != info["size"]:
                raise ValueError("Page image dimensions differ from manifest")
            return image.convert("RGB")
