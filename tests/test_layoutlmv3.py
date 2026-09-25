import json

import pytest
import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor,
                          LayoutLMv3Processor, LayoutLMv3Config, LayoutLMv3ForTokenClassification)

from app.data.schema import BBox, Word, Field, Document
from app.data.layoutlmv3_dataset import encode_layoutlmv3_windows, WINDOW_POLICY
from app.models_dl.layoutlmv3 import load_layoutlmv3_checkpoint
from app.data import SroieAdapter, CordAdapter


@pytest.fixture
def processor(tmp_path):
    vocab = {t: i for i, t in enumerate(['<s>', '<pad>', '</s>', '<unk>', '<mask>'] + sorted(ByteLevel.alphabet()))}
    (tmp_path / 'vocab.json').write_text(json.dumps(vocab))
    (tmp_path / 'merges.txt').write_text('#version: 0.2\n')
    tokenizer = LayoutLMv3TokenizerFast(vocab_file=str(tmp_path / 'vocab.json'), merges_file=str(tmp_path / 'merges.txt'))
    return LayoutLMv3Processor(LayoutLMv3ImageProcessor(apply_ocr=False, size={'height': 32, 'width': 32}), tokenizer)


def doc(doc_id='receipt', long=False):
    box = BBox(0, 0, .2, .2)
    words = [Word('noise', box, 0)] * (25 if long else 1) + [Word('TOTAL', box, 1)]
    return Document(doc_id, words, [Field('total', box, 1, 'TOTAL')], page_count=2)


def page_image(doc_id, page):
    return Image.new('RGB', (40, 60), 'black' if page == 0 else 'white')


def tiny_model(processor, vocabulary):
    cfg = LayoutLMv3Config(vocab_size=len(processor.tokenizer), hidden_size=36,
        coordinate_size=6, shape_size=6, num_attention_heads=4, num_hidden_layers=1,
        intermediate_size=72, input_size=32, patch_size=16, num_labels=len(vocabulary),
        label2id=vocabulary, id2label={i: v for v, i in vocabulary.items()})
    return LayoutLMv3ForTokenClassification(cfg)


def test_page_images_and_overflow_cover_each_word_once(processor):
    document = doc(long=True)
    windows = encode_layoutlmv3_windows(document, processor, {'O': 0, 'B-total': 1, 'I-total': 2}, page_image, max_length=17)
    assert [i for w in windows for i in w['word_indices'].tolist() if i >= 0] == list(range(len(document.words)))
    assert sum(int((w['labels'] == 1).sum()) for w in windows) == 1
    for w in windows:
        ids = [i for i in w['word_indices'].tolist() if i >= 0]
        page = document.words[ids[0]].page
        assert {document.words[i].page for i in ids} == {page}
        assert torch.all(w['pixel_values'] == (-1 if page == 0 else 1))
    assert encode_layoutlmv3_windows(Document('empty'), processor, {'O': 0}, page_image) == []
    with pytest.raises(ValueError, match='Missing page image'):
        encode_layoutlmv3_windows(document, processor, {'O': 0}, lambda *_: None)
    processor.image_processor.apply_ocr = True
    with pytest.raises(ValueError, match='apply_ocr=False'):
        encode_layoutlmv3_windows(document, processor, {'O': 0}, page_image)


def test_real_model_image_input_and_offline_reload(processor, tmp_path):
    torch.set_num_threads(1)
    vocabulary = {'O': 0, 'B-total': 1, 'I-total': 2}
    model = tiny_model(processor, vocabulary).eval()
    window = encode_layoutlmv3_windows(doc(), processor, vocabulary, page_image, max_length=16)[0]
    inputs = {k: window[k].unsqueeze(0) for k in ('input_ids', 'bbox', 'attention_mask', 'pixel_values')}
    with torch.no_grad():
        expected = model(**inputs).logits
        changed = model(**(inputs | {'pixel_values': -inputs['pixel_values']})).logits
    assert not torch.equal(expected, changed), 'Image input must affect the forward pass'
    path = tmp_path / 'best.pt'
    checkpoint = dict(model_type='layoutlmv3', window_policy=WINDOW_POLICY, config=model.config.to_dict(),
                      model_state_dict=model.state_dict(), vocabulary=vocabulary)
    torch.save(checkpoint, path)
    processor.save_pretrained(tmp_path / 'processor')
    loaded, _, loaded_vocab = load_layoutlmv3_checkpoint(path, torch.device('cpu'))
    with torch.no_grad():
        assert torch.allclose(loaded(**inputs).logits, expected)
    assert loaded_vocab == vocabulary
    checkpoint['window_policy'] = 'truncated'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='window policy'):
        load_layoutlmv3_checkpoint(path, torch.device('cpu'))


def test_receipt_adapters_refuse_missing_or_wrong_page_images(tmp_path):
    sroie = object.__new__(SroieAdapter)
    sroie.root = tmp_path
    with pytest.raises(FileNotFoundError):
        sroie.get_image('missing')
    with pytest.raises(ValueError, match='one page'):
        sroie.get_image('missing', 1)
    cord = object.__new__(CordAdapter)
    cord._dataset = {'train': [{'image': None}]}
    with pytest.raises(FileNotFoundError):
        cord.get_image('train_0')
    with pytest.raises(ValueError, match='one page'):
        cord.get_image('train_0', 1)


def test_tiny_training_reloads_and_exports_full_test(processor, tmp_path, monkeypatch):
    from app.training import train_layoutlmv3 as training
    from app.evaluation import predict
    torch.set_num_threads(1)
    vocabulary = {'O': 0, 'B-total': 1, 'I-total': 2}
    model_dir = tmp_path / 'source'
    tiny_model(processor, vocabulary).save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

    class Adapter:
        def splits(self): return ['train', 'val', 'test']
        def document_ids(self, split): return [split + '_0']
        def load(self, doc_id): return doc(doc_id)
        def iter_documents(self, split, limit=None): return iter([self.load(split + '_0')])
        get_image = staticmethod(page_image)

    adapter = Adapter()
    monkeypatch.setattr(training, 'load_adapter', lambda *_: adapter)
    monkeypatch.setattr(predict, 'load_adapter', lambda *_: adapter)
    training.main(['--dataset', 'sroie', '--model-path', str(model_dir), '--epochs', '1',
                   '--batch-size', '1', '--batch-accum', '3', '--device', 'cpu',
                   '--save-dir', str(tmp_path / 'checkpoints'), '--report-dir', str(tmp_path / 'reports')])
    metrics = json.loads((tmp_path / 'reports/predictions/predictions_layoutlmv3_sroie_test.metrics.json').read_text())
    assert metrics['complete'] and metrics['num_documents'] == 1
    assert metrics['window_policy'] == WINDOW_POLICY
    assert metrics['value_metrics']['num_documents'] == 1
    assert (tmp_path / 'checkpoints/layoutlmv3_sroie_latest.pt').is_file()
    report = json.loads((tmp_path / 'reports/metrics/layoutlmv3_sroie_results.json').read_text())
    assert report['best_epoch'] == 1 and report['split'] == 'TEST'
