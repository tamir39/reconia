import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from app.data.roberta_dataset import encode_document_windows, RoBERTaInvoiceDataset
from app.data.schema import BBox, Word, Field, Document
from app.evaluation.predict import predict_document_roberta
from app.training.train_roberta import evaluate_roberta_model


def fixture():
    base = Tokenizer(models.WordLevel({"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3,
                                       "noise": 4, "TOTAL": 5}, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.post_processor = processors.TemplateProcessing(single="<s> $A </s>",
                                                        special_tokens=[("<s>", 2), ("</s>", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", pad_token="<pad>")
    box = BBox(0, 0, .1, .1)
    words = [Word("noise", box, 0) for _ in range(600)] + [Word("TOTAL", box, 1)]
    doc = Document("long", words, [Field("total", box, 1, "TOTAL")], 2)
    vocab = {"O": 0, "B-total": 1, "I-total": 2}
    return tokenizer, doc, vocab


class TailModel(torch.nn.Module):
    def forward(self, input_ids, bbox, attention_mask=None, labels=None):
        logits = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
        logits[..., 0] = 1
        logits[..., 1] = (input_ids == 5).float() * 5
        return (logits.sum() * 0, logits) if labels is not None else logits


def test_windows_cover_every_word_once_and_keep_tail_entities():
    tokenizer, doc, vocab = fixture()
    windows = encode_document_windows(doc, tokenizer, vocab, max_seq_length=32)
    indices = [i for w in windows for i in w["word_indices"].tolist() if i >= 0]
    assert indices == list(range(601))
    assert sum((w["labels"] == 1).sum().item() for w in windows) == 1
    spans, predictions = predict_document_roberta(TailModel(), doc, tokenizer, vocab,
                                                  {v:k for k,v in vocab.items()}, torch.device("cpu"))
    assert len(spans) == 1
    assert (spans[0].start_token, spans[0].page, predictions[0]["text"]) == (600, 1, "TOTAL")
    result = evaluate_roberta_model(TailModel(), [doc, Document("empty")], tokenizer, vocab,
                                    {v:k for k,v in vocab.items()}, torch.device("cpu"), batch_size=2)
    assert result["f1"] == 1.0
    assert len(RoBERTaInvoiceDataset([Document("empty")], tokenizer, vocab)) == 0


def test_subword_continuation_at_window_boundary_is_not_counted_twice():
    base = Tokenizer(models.WordPiece({"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3,
                                       "invoice": 4, "##value": 5}, unk_token="[UNK]"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="[UNK]", pad_token="[PAD]")
    doc = Document("split-subword", [Word("invoicevalue", BBox(0, 0, .1, .1), 0) for _ in range(20)])
    windows = encode_document_windows(doc, tokenizer, {"O": 0}, max_seq_length=7)
    assert len(windows) > 1
    assert [i for w in windows for i in w["word_indices"].tolist() if i >= 0] == list(range(20))
    assert sum((w["labels"] != -100).sum().item() for w in windows) == 20
