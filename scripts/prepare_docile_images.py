"""Render and verify DocILE PDFs against the existing OCR export and fixed splits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

from app.data import DocileAdapter
from app.data.portable import ExportAdapter
from app.data.page_images import IMAGE_POLICY, PageImageStore, document_sha256, render_pdf, sha256


def prepare(root, export, output, executable, workers=4):
    raw = DocileAdapter(root=root)
    portable = ExportAdapter(export, "docile")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = {s: portable.document_ids(s) for s in ("train", "val", "test")}
    if splits != {s: raw.document_ids(s) for s in splits}:
        raise ValueError("Raw and exported splits differ")

    def one(doc_id):
        doc = portable.load(doc_id)
        if document_sha256(raw.load(doc_id)) != document_sha256(doc):
            raise ValueError(f"Raw/export document mismatch: {doc_id}")
        ann = raw._find_file("annotations", f"{doc_id}.json")
        ocr = raw._find_file("ocr", f"{doc_id}.json")
        pdf = raw.pdf_path(doc_id)
        metadata = json.loads(ann.read_bytes())["metadata"]
        ocr_pages = json.loads(ocr.read_bytes())["pages"]
        if metadata["page_count"] != doc.page_count or [p["page_idx"] for p in ocr_pages] != list(range(doc.page_count)):
            raise ValueError(f"Page count/index mismatch: {doc_id}")
        if any(not 0 <= item.page < doc.page_count for item in [*doc.words, *doc.fields]):
            raise ValueError(f"OCR/field page out of bounds: {doc_id}")
        source = dict(pdf_sha256=sha256(pdf), annotation_sha256=sha256(ann), ocr_sha256=sha256(ocr),
                      document_sha256=document_sha256(doc), policy=IMAGE_POLICY)
        destination = output / "pages" / doc_id
        record = destination / "record.json"
        if record.exists():
            old = json.loads(record.read_bytes())
            if all(old[k] == v for k, v in source.items()) and len(old["pages"]) == doc.page_count:
                if all(sha256(output / p["file"]) == p["sha256"] for p in old["pages"]):
                    return doc_id, old
            raise ValueError(f"Existing page bundle changed: {doc_id}; use a fresh destination")
        destination.mkdir(parents=True, exist_ok=True)
        pages = []
        with tempfile.TemporaryDirectory(prefix="render-", dir=output) as temporary:
            images = render_pdf(pdf, temporary, executable=executable)
            if len(images) != doc.page_count:
                raise ValueError(f"Rendered page count mismatch: {doc_id}")
            for index, path in enumerate(images):
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    expected_w, expected_h = metadata["page_sizes_at_200dpi"][index]
                    ocr_h, ocr_w = ocr_pages[index]["dimensions"]
                    ratio = image.width / image.height
                    if abs(ratio / (expected_w / expected_h) - 1) > .005 or abs(ratio / (ocr_w / ocr_h) - 1) > .005:
                        raise ValueError(f"Rendered/OCR aspect ratio mismatch: {doc_id}/{index}")
                    if any(w.page == index for w in doc.words) and all(lo == hi for lo, hi in image.getextrema()):
                        raise ValueError(f"Blank rendered page has OCR words: {doc_id}/{index}")
                    target = destination / f"{index:04d}.png"
                    image.save(target)
                    pages.append(dict(file=target.relative_to(output).as_posix(), sha256=sha256(target),
                                      size=list(image.size), source_size_200dpi=[expected_w, expected_h]))
        result = source | {"pages": pages}
        record.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return doc_id, result

    ids = [i for group in splits.values() for i in group]
    documents = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (doc_id, item) in enumerate(pool.map(one, ids), 1):
            documents[doc_id] = item
            if count % 100 == 0 or count == len(ids):
                print(f"Rendered and verified {count}/{len(ids)} documents", flush=True)
    manifest = dict(dataset="docile", policy=IMAGE_POLICY, splits=splits, documents=documents,
                    export_manifest_sha256=sha256(Path(export) / "manifest.json"),
                    renderer=subprocess.run([str(executable), "-v"], capture_output=True, text=True, check=True).stderr.splitlines()[0])
    (output / "image-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    PageImageStore(output, portable)
    print(json.dumps({"documents": len(ids), "pages": sum(len(d["pages"]) for d in documents.values()),
                      "image_manifest_sha256": sha256(output / "image-manifest.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdftoppm", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.root, args.export, args.output, args.pdftoppm, args.workers)


if __name__ == "__main__":
    main()
