import json
import shutil

from PIL import Image
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

from app.data.docile import DocileAdapter
from app.data.export import export_dataset
from app.data.page_images import IMAGE_POLICY, PageImageStore, document_sha256, sha256
from app.data.portable import ExportAdapter
from app.data.schema import Document, Word, BBox


def test_docile_renders_complete_vector_pages_and_rotation(tmp_path, monkeypatch):
    executable = shutil.which('pdftoppm')
    assert executable, 'Poppler must be installed to validate DocILE rendering'
    monkeypatch.setenv('PDFTOPPM', executable)
    (tmp_path / 'pdfs').mkdir()
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=100)
    content = DecodedStreamObject()
    content.set_data(b'1 0 0 rg 0 0 100 100 re f 0 0 1 rg 100 0 100 100 re f')
    page[NameObject('/Contents')] = writer._add_object(content)
    second = writer.add_blank_page(width=200, height=100)
    content = DecodedStreamObject()
    content.set_data(b'0 1 0 rg 0 0 200 100 re f')
    second[NameObject('/Contents')] = writer._add_object(content)
    second.rotate(90)
    writer.write(tmp_path / 'pdfs/vector.pdf')
    adapter = object.__new__(DocileAdapter)
    adapter.root, adapter.test_root = tmp_path, None
    image = adapter.get_image('vector', 0)
    assert image.size == (1600, 800)
    assert image.getpixel((100, 100)) == (255, 0, 0)
    assert image.getpixel((1500, 100)) == (0, 0, 255)
    rotated = adapter.get_image('vector', 1)
    assert rotated.size == (800, 1600)
    assert rotated.getpixel((100, 100)) == (0, 255, 0)
    for index in [-1, 2]:
        with pytest.raises(ValueError, match='outside PDF'):
            adapter.get_image('vector', index)
    with pytest.raises(FileNotFoundError):
        adapter.get_image('missing')


@pytest.fixture
def image_bundle(tmp_path):
    class Fixture:
        name = 'fixture'
        def splits(self):
            return ['train', 'val', 'test']
        def iter_documents(self, split):
            yield Document(split, [Word('invoice', BBox(0, 0, .5, .5), 1)], page_count=2)
    export = tmp_path / 'export'
    export_dataset(Fixture(), export, splits=('train', 'val', 'test'), shards=1)
    adapter = ExportAdapter(export, 'fixture')
    root = tmp_path / 'images'
    root.mkdir()
    documents = {}
    splits = {s: adapter.document_ids(s) for s in ('train', 'val', 'test')}
    for ids in splits.values():
        for doc_id in ids:
            pages = []
            for index, color in enumerate(['red', 'blue']):
                file = root / f'{doc_id}-{index}.png'
                Image.new('RGB', (20, 30), color).save(file)
                pages.append(dict(file=file.name, sha256=sha256(file), size=[20, 30]))
            documents[doc_id] = dict(document_sha256=document_sha256(adapter.load(doc_id)), pages=pages)
    manifest = dict(dataset='fixture', policy=IMAGE_POLICY, splits=splits, documents=documents)
    (root / 'image-manifest.json').write_text(json.dumps(manifest))
    return root, adapter, manifest


def test_verified_image_export_selects_page_and_rejects_corruption(image_bundle, monkeypatch):
    root, adapter, _ = image_bundle
    monkeypatch.setenv('FIXTURE_IMAGE_DIR', str(root))
    assert adapter.get_image('train', 1).getpixel((0, 0)) == (0, 0, 255)
    with pytest.raises(ValueError, match='Page index'):
        adapter.get_image('train', -1)
    (root / 'train-1.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        PageImageStore(root, adapter)


@pytest.mark.parametrize('change', ['ocr', 'page_count', 'split', 'path'])
def test_image_bundle_rejects_wrong_alignment(image_bundle, change):
    root, adapter, manifest = image_bundle
    if change == 'ocr':
        manifest['documents']['train']['document_sha256'] = 'incorrect'
    elif change == 'page_count':
        manifest['documents']['train']['pages'].pop()
    elif change == 'split':
        manifest['splits']['train'] = ['test']
    else:
        manifest['documents']['train']['pages'][0]['file'] = '../outside.png'
    (root / 'image-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        PageImageStore(root, adapter)
