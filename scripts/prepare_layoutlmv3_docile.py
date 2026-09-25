"""Build an allowlisted private Kaggle bundle with verified DocILE page images."""

import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from app.data.page_images import PageImageStore, sha256
from app.data.portable import ExportAdapter

ROOT = Path(__file__).resolve().parents[1]


def prepare(export, images, output, owner):
    export, images, output = Path(export), Path(images), Path(output)
    adapter = ExportAdapter(export, 'docile')
    store = PageImageStore(images, adapter)
    if {s: len(adapter.document_ids(s)) for s in ('train', 'val', 'test')} != {'train': 4680, 'val': 500, 'test': 500}:
        raise ValueError('Expected the existing complete DocILE split')
    if sha256(export / 'manifest.json') != store.manifest['export_manifest_sha256']:
        raise ValueError('Images were prepared against a different OCR export')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Choose a fresh upload bundle directory')
    sources = [p for p in (ROOT / 'app').rglob('*')
               if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts]
    sources += [ROOT / 'docs/EVALUATION.md', ROOT / 'docs/LAYOUTLMV3.md', ROOT / 'docs/DOCILE_IMAGES.md']
    provenance = dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      files_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in sources})
    with zipfile.ZipFile(output / 'reconia-code.bin', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('source-provenance.json', json.dumps(provenance, indent=2))
    manifest = json.loads((export / 'manifest.json').read_bytes())
    with zipfile.ZipFile(output / 'docile-export.bin', 'w', zipfile.ZIP_STORED) as archive:
        for name in ['manifest.json', *manifest['checksums']]:
            path = (export / name).resolve()
            if not path.is_relative_to(export.resolve()):
                raise ValueError('Export path escapes directory')
            archive.write(path, name)
    with zipfile.ZipFile(output / 'docile-images.bin', 'w', zipfile.ZIP_STORED) as archive:
        archive.write(images / 'image-manifest.json', 'image-manifest.json')
        for doc in store.manifest['documents'].values():
            for page in doc['pages']:
                archive.write(images / page['file'], page['file'])
    checksums = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in output.glob('*.bin')}
    (output / 'bundle-manifest.json').write_text(json.dumps(checksums, indent=2) + '\n', encoding='utf-8')
    metadata = dict(title='Reconia DocILE LayoutLMv3 Training Bundle',
                    id=f'{owner}/reconia-docile-layoutlmv3-training-bundle',
                    licenses=[{'name': 'other'}],
                    description='Private research input: DocILE OCR/labels and all 7394 rendered PDF pages for the fixed 4680/500/500 internal split. Complete pages, lossless RGB PNG, longest edge 1600; existing OCR is retained. Derived from https://docile.rossum.ai/ and https://github.com/rossumai/docile. Original rights remain with the authors; this upload does not relicense DocILE. Includes allowlisted Reconia source and hashes. No credentials, original PDFs or trained weights.')
    (output / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checksums, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['export', 'images', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    prepare(args.export, args.images, args.output, args.owner)


if __name__ == '__main__':
    main()
