from app.data.export import read_shards
from app.data.schema import Document, Word, Field, BBox
from app.graph.labels import label_document
from scripts.prepare_lilt import export_receipt_lossless


def test_receipt_export_keeps_full_coordinate_and_confidence_precision(tmp_path):
    box = BBox(59 / 576, 485 / 864, 209 / 576, 543 / 864)
    document = Document('receipt', [Word('amount', box, 0, confidence=0.987654321)],
                        [Field('total', box, 0, 'amount')])

    class Adapter:
        name = 'synthetic'
        def splits(self): return ['train', 'val', 'test']
        def load(self, _): return document
        def iter_documents(self, split): return iter([document])

    export_receipt_lossless(Adapter(), tmp_path)
    copy = next(read_shards(tmp_path, 'train'))
    assert copy.words == document.words
    assert copy.fields == document.fields
    assert label_document(copy) == label_document(document)
