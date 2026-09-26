"""Package verified OCR exports and allowlisted LiLT source for private Kaggle use."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from app.data import SroieAdapter, CordAdapter
from app.data.export import export_dataset, document_to_dict
from app.data.portable import ExportAdapter
from app.data.page_images import sha256
from app.graph.labels import label_document

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {'docile': {'train': 4680, 'val': 500, 'test': 500},
          'sroie': {'train': 500, 'val': 63, 'test': 63},
          'cord': {'train': 800, 'val': 100, 'test': 100}}


def export_receipt_lossless(adapter, target):
    """Keep adapter coordinates exact: rounding can change BIO overlap thresholds."""
    manifest = export_dataset(adapter, target, splits=('train', 'val', 'test'))
    for name in manifest['checksums']:
        path = target / name
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            ids = [json.loads(line)['doc_id'] for line in stream]
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for doc_id in ids:
                doc = adapter.load(doc_id)
                payload = document_to_dict(doc)
                for source, item in zip(doc.words, payload['words']):
                    item['b'] = [source.bbox.x0, source.bbox.y0, source.bbox.x1, source.bbox.y1]
                    if source.confidence is not None:
                        item['c'] = source.confidence
                for source, item in zip(doc.fields, payload['fields']):
                    item['b'] = [source.bbox.x0, source.bbox.y0, source.bbox.x1, source.bbox.y1]
                stream.write(json.dumps(payload, ensure_ascii=False)+'\n')
        manifest['checksums'][name] = sha256(path)
    manifest['coordinate_precision'] = 'full-adapter-precision'
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def prepare(docile_export, sroie, cord, output, owner):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a fresh output directory')
    upload = output / 'upload'
    upload.mkdir(parents=True)
    exports = {'docile': Path(docile_export)}
    receipt_proof = {}
    for name, cls, root in [('sroie', SroieAdapter, sroie), ('cord', CordAdapter, cord)]:
        adapter = cls(root=Path(root))
        target = output / name
        export_receipt_lossless(adapter, target)
        exported = ExportAdapter(target, name)
        for split in COUNTS[name]:
            # CORD calls its validation split "val" in its local adapter.
            assert set(adapter.document_ids(split)) == set(exported.document_ids(split))
            for doc_id in exported.document_ids(split):
                original, copy = adapter.load(doc_id), exported.load(doc_id)
                assert document_to_dict(original) == document_to_dict(copy), doc_id
                assert original.words == copy.words and original.fields == copy.fields, doc_id
                assert label_document(original)[0] == label_document(copy)[0], doc_id
        exports[name] = target
        receipt_proof[name] = {'all_documents_match': True, 'bio_labels_unchanged': True}
    for name, directory in exports.items():
        adapter = ExportAdapter(directory, name)
        assert {s: len(adapter.document_ids(s)) for s in COUNTS[name]} == COUNTS[name]
        manifest = json.loads((directory / 'manifest.json').read_bytes())
        with zipfile.ZipFile(upload / f'{name}-export.bin', 'w', zipfile.ZIP_STORED) as archive:
            for file in ['manifest.json', *manifest['checksums']]:
                path = (directory / file).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError('Manifest path escapes export')
                archive.write(path, file)
    shutil.copyfile(Path(sroie) / 'field-value-targets.json', upload / 'sroie-field-value-targets.json')
    sources = [p for p in (ROOT / 'app').rglob('*')
               if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts]
    sources += [ROOT / 'docs/EVALUATION.md', ROOT / 'docs/LILT.md']
    provenance = dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      files_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in sources})
    with zipfile.ZipFile(upload / 'reconia-code.bin', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('source-provenance.json', json.dumps(provenance, indent=2))
    manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in upload.iterdir()}
    (upload / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    metadata = dict(title='Reconia LiLT Training Bundle', id=f'{owner}/reconia-lilt-training-bundle',
        licenses=[{'name': 'other'}], description='Private research bundle: fixed DocILE/SROIE/CORD OCR, boxes and labels, SROIE original-key targets, and allowlisted Reconia source. No images, secrets or trained weights. Original dataset rights remain with their authors; this upload does not relicense data.')
    (upload / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    (output / 'verification.json').write_text(json.dumps(dict(counts=COUNTS,
        receipt_export_verification=receipt_proof, files=manifest), indent=2)+'\n', encoding='utf-8')
    print('Verified bundle prepared:', upload, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['docile-export', 'sroie', 'cord', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    prepare(args.docile_export, args.sroie, args.cord, args.output, args.owner)
