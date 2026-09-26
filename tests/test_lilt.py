import json

import pytest
import torch
from tokenizers.pre_tokenizers import ByteLevel
from transformers import LayoutLMv3TokenizerFast, LiltConfig, LiltForTokenClassification

from app.data.schema import BBox, Word, Field, Document
from app.data.lilt_dataset import encode_lilt_windows, WINDOW_POLICY
from app.models_dl.lilt import load_lilt_checkpoint


@pytest.fixture
def tokenizer(tmp_path):
    vocab = {t: i for i, t in enumerate(['<s>', '<pad>', '</s>', '<unk>', '<mask>'] + sorted(ByteLevel.alphabet()))}
    (tmp_path / 'vocab.json').write_text(json.dumps(vocab))
    (tmp_path / 'merges.txt').write_text('#version: 0.2\n')
    tokenizer = LayoutLMv3TokenizerFast(vocab_file=str(tmp_path / 'vocab.json'), merges_file=str(tmp_path / 'merges.txt'))
    return tokenizer


def doc(doc_id='receipt', long=False):
    box = BBox(0, 0, .2, .2)
    words = [Word('noise', box, 0)] * (25 if long else 1) + [Word('TOTAL', box, 1)]
    return Document(doc_id, words, [Field('total', box, 1, 'TOTAL')], page_count=2)


def tiny_model(tokenizer, vocabulary):
    cfg = LiltConfig(vocab_size=len(tokenizer), hidden_size=48,
        channel_shrink_ratio=2, num_attention_heads=4, num_hidden_layers=1,
        intermediate_size=96, num_labels=len(vocabulary),
        label2id=vocabulary, id2label={i: v for v, i in vocabulary.items()})
    return LiltForTokenClassification(cfg)


def test_pages_and_overflow_cover_each_word_once(tokenizer):
    document = doc(long=True)
    windows = encode_lilt_windows(document, tokenizer, {'O': 0, 'B-total': 1, 'I-total': 2}, max_length=17)
    assert [i for w in windows for i in w['word_indices'].tolist() if i >= 0] == list(range(len(document.words)))
    assert sum(int((w['labels'] == 1).sum()) for w in windows) == 1
    for w in windows:
        ids = [i for i in w['word_indices'].tolist() if i >= 0]
        page = document.words[ids[0]].page
        assert {document.words[i].page for i in ids} == {page}
        assert "pixel_values" not in w
    assert encode_lilt_windows(Document('empty'), tokenizer, {'O': 0}) == []


def test_real_model_layout_input_and_offline_reload(tokenizer, tmp_path):
    torch.set_num_threads(1)
    vocabulary = {'O': 0, 'B-total': 1, 'I-total': 2}
    model = tiny_model(tokenizer, vocabulary).eval()
    window = encode_lilt_windows(doc(), tokenizer, vocabulary, max_length=16)[0]
    inputs = {k: window[k].unsqueeze(0) for k in ('input_ids', 'bbox', 'attention_mask')}
    with torch.no_grad():
        expected = model(**inputs).logits
        changed = model(**(inputs | {'bbox': (inputs['bbox'] + 100).clamp(max=1000)})).logits
    assert not torch.equal(expected, changed), 'Layout input must affect the forward pass'
    path = tmp_path / 'best.pt'
    checkpoint = dict(model_type='lilt', window_policy=WINDOW_POLICY, config=model.config.to_dict(),
                      model_state_dict=model.state_dict(), vocabulary=vocabulary)
    torch.save(checkpoint, path)
    tokenizer.save_pretrained(tmp_path / 'tokenizer')
    loaded, _, loaded_vocab = load_lilt_checkpoint(path, torch.device('cpu'))
    with torch.no_grad():
        assert torch.allclose(loaded(**inputs).logits, expected)
    assert loaded_vocab == vocabulary
    checkpoint['window_policy'] = 'truncated'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='window policy'):
        load_lilt_checkpoint(path, torch.device('cpu'))


@pytest.mark.parametrize('dataset', ['sroie', 'cord', 'docile'])
def test_tiny_training_reloads_and_exports_full_test(tokenizer, tmp_path, monkeypatch, dataset):
    from app.training import train_lilt as training
    from app.evaluation import predict
    torch.set_num_threads(1)
    vocabulary = {'O': 0, 'B-total': 1, 'I-total': 2}
    model_dir = tmp_path / 'source'
    tiny_model(tokenizer, vocabulary).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    class Adapter:
        def splits(self): return ['train', 'val', 'test']
        def document_ids(self, split): return [split + '_0']
        def load(self, doc_id): return doc(doc_id)
        def iter_documents(self, split, limit=None): return iter([self.load(split + '_0')])

    adapter = Adapter()
    monkeypatch.setattr(training, 'load_adapter', lambda *_: adapter)
    monkeypatch.setattr(predict, 'load_adapter', lambda *_: adapter)
    training.main(['--dataset', dataset, '--model-path', str(model_dir), '--epochs', '1',
                   '--batch-size', '1', '--batch-accum', '3', '--device', 'cpu',
                   '--save-dir', str(tmp_path / 'checkpoints'), '--report-dir', str(tmp_path / 'reports')])
    metrics = json.loads((tmp_path / f'reports/predictions/predictions_lilt_{dataset}_test.metrics.json').read_text())
    assert metrics['complete'] and metrics['num_documents'] == 1
    assert metrics['window_policy'] == WINDOW_POLICY
    assert metrics['value_metrics']['num_documents'] == 1
    assert (tmp_path / f'checkpoints/lilt_{dataset}_latest.pt').is_file()
    report = json.loads((tmp_path / f'reports/metrics/lilt_{dataset}_results.json').read_text())
    assert report['best_epoch'] == 1 and report['split'] == 'TEST'
