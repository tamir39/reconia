"""Spatial Relational Graph Neural Network for Document Information Extraction.

Implements the multi-modal graph convolutional architecture grounded in the seminal works:
  - Liu et al. (2019): Graph Convolution for Multimodal Information Extraction from Visually Rich Documents.
  - PICK (Yu et al., 2020): Processing Key Information Extraction from Documents.

Key Architecture Components:
  1. Node Embedding:
     - Semantic Text Encoder: Subword/character hash embedding bag mapping text to d_text.
     - Geometric Layout: 17 normalized layout features.
     - Node input = [h_text || h_geom] (dimension: d_text + 17).
  2. Edge Representation:
     - Discrete edge type embedding (same-line, vertical-nearest, knn, self-loop).
     - Continuous spatial relation vector r_ij in R^6: [dx, dy, w_i/h_i, h_j/h_i, w_j/h_i, dist_ij].
  3. Message Passing with Spatial Attention:
     - Relational message calculation conditioned on edge spatial relations.
     - LeakyReLU spatial attention weights normalized via edge softmax.
     - Residual connections, LayerNorm, and Dropout.
  4. Classification Head:
     - Multi-class BIO node classification for KILE / LIR.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

TEXT_HASH_VERSION = "blake2b-64-v1"


def validate_gnn_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Refuse weights trained against Python's process-dependent hash table."""
    if checkpoint.get("text_hash_version") != TEXT_HASH_VERSION:
        raise ValueError(
            "This GNN checkpoint uses an unversioned or incompatible text hash. "
            "Retrain with the stable text encoder; legacy weights cannot be "
            "safely converted without their original hash mapping."
        )

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    # Allow imports when torch is not yet installed or in pure-python environments
    torch = None
    nn = object
    F = None


@dataclass
class InvoiceGNNConfig:
    """Configuration for InvoiceGNN."""

    geo_dim: int = 17
    text_dim: int = 128
    edge_spatial_dim: int = 6
    num_edge_types: int = 4  # same_line (0), vertical (1), knn (2), self_loop (3)
    edge_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 9
    dropout: float = 0.2
    hash_vocab_size: int = 30000


if torch is not None:

    class SubwordTextEncoder(nn.Module):
        """Encodes variable-length word strings into fixed-size semantic vectors using character n-gram hashing."""

        def __init__(self, vocab_size: int = 30000, embedding_dim: int = 128) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embedding = nn.EmbeddingBag(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                mode="mean",
            )
            # Initialize embeddings with standard normal
            nn.init.normal_(self.embedding.weight, std=0.05)

        def _hash_string(self, text: str) -> list[int]:
            """Extract character 3-grams and hash them deterministically."""
            clean = f"<{text.strip().lower()}>"
            ngrams = [clean] if len(clean) < 3 else [clean[i:i + 3] for i in range(len(clean) - 2)]
            return [
                int.from_bytes(hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest(), "big")
                % self.vocab_size
                for ngram in ngrams
            ]

        def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
            """Encode a list of N strings into a tensor of shape (N, embedding_dim)."""
            if not texts:
                return torch.zeros((0, self.embedding_dim), device=device)

            indices: list[int] = []
            offsets: list[int] = []
            curr_offset = 0

            for t in texts:
                ngram_hashes = self._hash_string(t)
                offsets.append(curr_offset)
                indices.extend(ngram_hashes)
                curr_offset += len(ngram_hashes)

            indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            offsets_tensor = torch.tensor(offsets, dtype=torch.long, device=device)

            return self.embedding(indices_tensor, offsets_tensor)

    class SpatialEdgeEncoder(nn.Module):
        """Encodes discrete edge types and continuous spatial geometry vectors into unified edge embeddings."""

        def __init__(
            self,
            num_edge_types: int = 4,
            spatial_dim: int = 6,
            edge_dim: int = 64,
        ) -> None:
            super().__init__()
            self.type_embed = nn.Embedding(num_edge_types, edge_dim)
            self.spatial_proj = nn.Sequential(
                nn.Linear(spatial_dim, edge_dim),
                nn.LayerNorm(edge_dim),
                nn.GELU(),
            )
            self.combine = nn.Linear(edge_dim * 2, edge_dim)

        def forward(
            self, edge_type: torch.Tensor, edge_attr: torch.Tensor
        ) -> torch.Tensor:
            """Args:

            edge_type: (E,) int64 edge_attr: (E, spatial_dim) float32
            Returns:
                (E, edge_dim) float32
            """
            type_emb = self.type_embed(edge_type)
            spatial_emb = self.spatial_proj(edge_attr)
            return self.combine(torch.cat([type_emb, spatial_emb], dim=-1))

    class SpatialGNNLayer(nn.Module):
        """Single spatial relational graph convolution layer with edge attention and residual connection."""

        def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            out_dim: int,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.out_dim = out_dim

            # Message transform: combines source node features with edge features
            self.msg_linear = nn.Linear(node_dim + edge_dim, out_dim)

            # Spatial attention: conditioned on target node and incoming message
            self.attn_linear = nn.Linear(node_dim + out_dim, 1)

            self.norm = nn.LayerNorm(out_dim)
            self.dropout = nn.Dropout(dropout)
            self.shortcut = (
                nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()
            )

        def forward(
            self,
            h: torch.Tensor,
            edge_index: torch.Tensor,
            edge_emb: torch.Tensor,
        ) -> torch.Tensor:
            """Args:

            h: (N, node_dim) node representations edge_index: (2, E) where
            edge_index[0] is source j, edge_index[1] is target i edge_emb: (E,
            edge_dim) edge embeddings Returns:
                (N, out_dim) updated node representations
            """
            num_nodes = h.size(0)
            num_edges = edge_index.size(1)

            if num_edges == 0 or num_nodes == 0:
                return self.norm(self.shortcut(h))

            src, dst = edge_index[0], edge_index[1]

            # 1. Compute edge messages m_ij = GELU(W_m [h_j || e_ij])
            src_h = h[src]  # (E, node_dim)
            msg_input = torch.cat([src_h, edge_emb], dim=-1)
            messages = F.gelu(self.msg_linear(msg_input))  # (E, out_dim)

            # 2. Compute attention coefficients
            dst_h = h[dst]  # (E, node_dim)
            attn_input = torch.cat([dst_h, messages], dim=-1)
            scores = F.leaky_relu(self.attn_linear(attn_input), negative_slope=0.2)  # (E, 1)

            # Numerically stable edge softmax grouped by dst node:
            # We compute exp(scores - max) / sum(exp(scores - max))
            max_scores = torch.full((num_nodes, 1), -1e9, device=h.device, dtype=h.dtype)
            max_scores.scatter_reduce_(
                0, dst.unsqueeze(-1), scores, reduce="amax", include_self=False
            )
            # Clamp to prevent NaN when a node has no incoming edges
            max_per_edge = max_scores[dst]
            exp_scores = torch.exp(scores - max_per_edge)  # (E, 1)

            sum_exp = torch.zeros((num_nodes, 1), device=h.device, dtype=h.dtype)
            sum_exp.scatter_add_(0, dst.unsqueeze(-1), exp_scores)
            sum_per_edge = sum_exp[dst] + 1e-8

            alpha = exp_scores / sum_per_edge  # (E, 1)
            weighted_messages = alpha * messages  # (E, out_dim)

            # 3. Aggregate messages into target nodes
            aggregated = torch.zeros((num_nodes, self.out_dim), device=h.device, dtype=h.dtype)
            aggregated.index_add_(0, dst, weighted_messages)

            # 4. Residual connection, dropout, and layer normalization
            out = self.norm(self.shortcut(h) + self.dropout(aggregated))
            return out

    class InvoiceGNN(nn.Module):
        """End-to-end multi-modal Graph Neural Network for invoice token classification."""

        def __init__(self, config: InvoiceGNNConfig) -> None:
            super().__init__()
            self.config = config

            # 1. Semantic text encoder
            self.text_encoder = SubwordTextEncoder(
                vocab_size=config.hash_vocab_size,
                embedding_dim=config.text_dim,
            )

            # 2. Spatial edge encoder
            self.edge_encoder = SpatialEdgeEncoder(
                num_edge_types=config.num_edge_types,
                spatial_dim=config.edge_spatial_dim,
                edge_dim=config.edge_dim,
            )

            # 3. Node feature projection: [text_dim + geo_dim] -> hidden_dim
            in_node_dim = config.text_dim + config.geo_dim
            self.node_proj = nn.Sequential(
                nn.Linear(in_node_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )

            # 4. Stack of Spatial Relational GNN layers
            self.gnn_layers = nn.ModuleList([
                SpatialGNNLayer(
                    node_dim=config.hidden_dim,
                    edge_dim=config.edge_dim,
                    out_dim=config.hidden_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ])

            # 5. Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.num_classes),
            )

        def forward(
            self,
            geo_features: torch.Tensor,
            texts: list[str],
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> torch.Tensor:
            """Args:

            geo_features: (N, 17) float32 layout features texts: list of N
            string tokens edge_index: (2, E) int64 undirected or directed edge
            pairs edge_type: (E,) int64 edge relation types edge_attr: (E, 6)
            float32 spatial relation vectors Returns:
                logits: (N, num_classes) unnormalized log-probabilities per node
            """
            device = geo_features.device
            num_nodes = geo_features.size(0)

            # 1. Encode text semantics
            text_emb = self.text_encoder(texts, device=device)  # (N, text_dim)

            # 2. Concatenate [text || geometry] and project
            node_feat = torch.cat([text_emb, geo_features], dim=-1)  # (N, text_dim + 17)
            h = self.node_proj(node_feat)  # (N, hidden_dim)

            # 3. Encode edge features
            edge_emb = self.edge_encoder(edge_type, edge_attr)  # (E, edge_dim)

            # 4. Message passing through GNN layers
            for layer in self.gnn_layers:
                h = layer(h, edge_index, edge_emb)

            # 5. Classification head
            logits = self.classifier(h)  # (N, num_classes)
            return logits

else:
    # Dummy classes for environments where torch is not installed
    class InvoiceGNN:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required to use InvoiceGNN.")
