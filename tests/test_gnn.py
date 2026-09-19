"""Unit tests for InvoiceGNN architecture and components."""

import pytest
import numpy as np

# Skip test suite if torch is not yet installed
torch = pytest.importorskip("torch")
import torch.nn as nn

from app.models_dl.gnn import (
    InvoiceGNN,
    InvoiceGNNConfig,
    SpatialEdgeEncoder,
    SpatialGNNLayer,
    SubwordTextEncoder,
)


def test_subword_text_encoder():
    encoder = SubwordTextEncoder(vocab_size=1000, embedding_dim=64)
    texts = ["TOTAL", "RM120.00", "2023-01-01", "A"]
    device = torch.device("cpu")

    emb = encoder(texts, device=device)
    assert emb.shape == (4, 64)
    assert not torch.isnan(emb).any()


def test_spatial_edge_encoder():
    encoder = SpatialEdgeEncoder(num_edge_types=4, spatial_dim=6, edge_dim=32)
    edge_type = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    edge_attr = torch.randn(4, 6, dtype=torch.float32)

    emb = encoder(edge_type, edge_attr)
    assert emb.shape == (4, 32)
    assert not torch.isnan(emb).any()


def test_spatial_gnn_layer():
    layer = SpatialGNNLayer(node_dim=64, edge_dim=32, out_dim=64, dropout=0.1)

    # 3 nodes: 0, 1, 2
    h = torch.randn(3, 64)
    # Directed edges: (0 -> 1), (1 -> 2), (2 -> 0)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_emb = torch.randn(3, 32)

    out = layer(h, edge_index, edge_emb)
    assert out.shape == (3, 64)
    assert not torch.isnan(out).any()


def test_invoice_gnn_forward_and_backward():
    config = InvoiceGNNConfig(
        geo_dim=17,
        text_dim=32,
        edge_spatial_dim=6,
        num_edge_types=4,
        edge_dim=16,
        hidden_dim=32,
        num_layers=2,
        num_classes=5,
        dropout=0.1,
    )
    model = InvoiceGNN(config)

    # 4 nodes
    geo_features = torch.randn(4, 17)
    texts = ["Invoice", "No:", "12345", "Total"]
    # Edges: 0 <-> 1, 1 <-> 2, 2 <-> 3 (6 directed edges)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    edge_attr = torch.randn(6, 6)

    # Forward
    logits = model(geo_features, texts, edge_index, edge_type, edge_attr)
    assert logits.shape == (4, 5)
    assert not torch.isnan(logits).any()

    # Backward
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss.backward()

    # Verify gradients computed
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
            has_grad = True
    assert has_grad

    optimizer.step()
