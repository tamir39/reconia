"""Unit tests for RoBERTa with 2D Layout Positional Embedding (Model 2)."""

import pytest
import numpy as np

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import RobertaConfig, RobertaModel, AutoTokenizer

from app.data.schema import BBox, Document, Word, Field
from app.data.roberta_dataset import (
    RoBERTaInvoiceDataset,
    encode_document_for_roberta,
    extract_word_bboxes_and_labels,
)
from app.models_dl.roberta import (
    LayoutEmbedding,
    RoBERTaWithLayoutConfig,
    RoBERTaWithLayoutForTokenClassification,
)


def test_layout_embedding():
    layer = LayoutEmbedding(hidden_size=64, max_2d_position_embeddings=1001, dropout=0.0)
    # Shape: (batch_size=2, seq_len=4, coords=4)
    bbox = torch.tensor(
        [
            [[0, 0, 100, 100], [200, 300, 400, 500], [900, 900, 1000, 1000], [0, 0, 0, 0]],
            [[10, 20, 30, 40], [50, 60, 70, 80], [100, 200, 300, 400], [500, 600, 700, 800]],
        ],
        dtype=torch.long,
    )
    emb = layer(bbox)
    assert emb.shape == (2, 4, 64)
    assert not torch.isnan(emb).any()


def test_layout_embedding_clamping():
    layer = LayoutEmbedding(hidden_size=32, max_2d_position_embeddings=1001)
    # Coordinates exceeding 1000 or negative should be safely clamped
    bbox = torch.tensor([[[-50, -10, 1500, 2000]]], dtype=torch.long)
    emb = layer(bbox)
    assert emb.shape == (1, 1, 32)
    assert not torch.isnan(emb).any()


def test_roberta_with_layout_forward_and_backward():
    # Use lightweight config for fast unit testing
    small_cfg = RobertaConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=514,
    )
    small_roberta = RobertaModel(small_cfg)
    custom_cfg = RoBERTaWithLayoutConfig(
        num_labels=7,
        hidden_size=64,
        classifier_dropout=0.1,
    )
    model = RoBERTaWithLayoutForTokenClassification(
        config=custom_cfg,
        roberta_model=small_roberta,
    )

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    bbox = torch.randint(0, 1000, (batch_size, seq_len, 4))
    attention_mask = torch.ones((batch_size, seq_len))
    labels = torch.randint(0, 7, (batch_size, seq_len))
    # Add some -100 ignore tokens
    labels[0, 0] = -100
    labels[0, 5:] = -100

    # 1. Forward with labels
    loss, logits = model(input_ids, bbox, attention_mask, labels=labels)
    assert logits.shape == (batch_size, seq_len, 7)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss).any()
    assert not torch.isnan(logits).any()

    # 2. Forward without labels
    logits_only = model(input_ids, bbox, attention_mask)
    assert logits_only.shape == (batch_size, seq_len, 7)

    # 3. Backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
            has_grad = True
    assert has_grad
    optimizer.step()


def test_roberta_dataset_encoding(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    label_vocab = {"O": 0, "B-total": 1, "I-total": 2, "B-date": 3, "I-date": 4}

    doc = Document(
        doc_id="test_doc_01",
        words=[
            Word(text="TOTAL", bbox=BBox(0.1, 0.8, 0.3, 0.85), page=0),
            Word(text="RM150.00", bbox=BBox(0.35, 0.8, 0.6, 0.85), page=0),
            Word(text="2023-10-15", bbox=BBox(0.1, 0.2, 0.4, 0.25), page=0),
        ],
        fields=[
            Field(fieldtype="total", bbox=BBox(0.35, 0.8, 0.6, 0.85), page=0, text="RM150.00"),
            Field(fieldtype="date", bbox=BBox(0.1, 0.2, 0.4, 0.25), page=0, text="2023-10-15"),
        ],
    )

    encoded = encode_document_for_roberta(
        document=doc,
        tokenizer=tokenizer,
        label_vocabulary=label_vocab,
        max_seq_length=32,
    )

    assert "input_ids" in encoded
    assert "bbox" in encoded
    assert "attention_mask" in encoded
    assert "labels" in encoded

    assert encoded["input_ids"].shape == (32,)
    assert encoded["bbox"].shape == (32, 4)
    assert encoded["labels"].shape == (32,)

    # Special token <s> is at index 0, should have bbox [0, 0, 0, 0] and label -100
    assert encoded["labels"][0].item() == -100
    assert (encoded["bbox"][0] == 0).all()

    # Verify at least some valid tokens have non-zero boxes and non -100 labels
    valid_labels = encoded["labels"][encoded["labels"] != -100]
    assert len(valid_labels) > 0


def test_roberta_model_serialization(tmp_path):
    small_cfg = RobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=128,
    )
    small_roberta = RobertaModel(small_cfg)
    cfg = RoBERTaWithLayoutConfig(
        num_labels=3,
        hidden_size=32,
    )
    model = RoBERTaWithLayoutForTokenClassification(config=cfg, roberta_model=small_roberta)

    save_dir = tmp_path / "roberta_ckpt"
    model.save_pretrained(save_dir)

    assert (save_dir / "reconia_roberta_config.json").exists()
    assert (save_dir / "model_weights.pt").exists()
