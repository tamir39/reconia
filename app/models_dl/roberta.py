"""RoBERTa-base with 2D Layout Positional Embedding for Token Classification.

Model 2 in RECONIA benchmark specification (docs/SPEC.md §4.2):
Combines sequential pre-trained language modeling (RoBERTa-base) with 2D spatial
layout embeddings (x0, y0, x1, y1, width, height) inspired by LAMBERT
(Garncarek et al., 2021) and the DocILE benchmark baseline (Simsa et al., ICDAR 2023).

Acts as the 1D+2D non-visual baseline to compare against GNNs and multimodal
models (LayoutLMv3, LiLT).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel


DEFAULT_LOCAL_PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "pretrained" / "roberta-base"


def get_default_roberta_model_path() -> str:
    """Return local pre-trained roberta-base path if present, otherwise 'roberta-base'."""
    if DEFAULT_LOCAL_PRETRAINED_DIR.exists() and (DEFAULT_LOCAL_PRETRAINED_DIR / "config.json").exists():
        return str(DEFAULT_LOCAL_PRETRAINED_DIR)
    return "roberta-base"


@dataclass
class RoBERTaWithLayoutConfig:
    """Configuration for RoBERTaWithLayoutForTokenClassification."""

    model_name_or_path: str = "roberta-base"
    num_labels: int = 9
    max_2d_position_embeddings: int = 1001
    hidden_size: int = 768
    classifier_dropout: float = 0.1
    label2id: dict[str, int] | None = None
    id2label: dict[int, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoBERTaWithLayoutConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LayoutEmbedding(nn.Module):
    """2D spatial layout embedding layer for bounding boxes in [0, 1000].

    Embeds [x0, y0, x1, y1, width, height] coordinates into 6 discrete embedding
    tables and sums them with LayerNorm and Dropout, matching LAMBERT and LayoutLM.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        max_2d_position_embeddings: int = 1001,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_coord = max_2d_position_embeddings - 1
        self.x0_emb = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.y0_emb = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.x1_emb = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.y1_emb = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.w_emb = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.h_emb = nn.Embedding(max_2d_position_embeddings, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """Forward pass for 2D bounding boxes.

        Args:
            bbox: Tensor of shape (batch_size, seq_len, 4) with [x0, y0, x1, y1]
                  normalized to integers in [0, 1000].

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Clamp coordinates to valid range [0, max_coord]
        x0 = bbox[:, :, 0].clamp(0, self.max_coord)
        y0 = bbox[:, :, 1].clamp(0, self.max_coord)
        x1 = bbox[:, :, 2].clamp(0, self.max_coord)
        y1 = bbox[:, :, 3].clamp(0, self.max_coord)

        w = (x1 - x0).clamp(0, self.max_coord)
        h = (y1 - y0).clamp(0, self.max_coord)

        spatial_emb = (
            self.x0_emb(x0)
            + self.y0_emb(y0)
            + self.x1_emb(x1)
            + self.y1_emb(y1)
            + self.w_emb(w)
            + self.h_emb(h)
        )
        return self.dropout(spatial_emb)


class RoBERTaWithLayoutForTokenClassification(nn.Module):
    """RoBERTa model with integrated 2D spatial layout embeddings for BIO token classification."""

    def __init__(
        self,
        config: RoBERTaWithLayoutConfig | None = None,
        roberta_model: RobertaModel | None = None,
        allow_random_init: bool = False,
    ) -> None:
        super().__init__()
        self.custom_config = config or RoBERTaWithLayoutConfig()
        self.num_labels = self.custom_config.num_labels

        if roberta_model is not None:
            self.roberta = roberta_model
        else:
            model_source = self.custom_config.model_name_or_path
            if model_source == "roberta-base":
                model_source = get_default_roberta_model_path()

            try:
                self.roberta = RobertaModel.from_pretrained(model_source)
                print(f"[RoBERTa] Successfully loaded pre-trained weights from '{model_source}'.")
            except Exception as err:
                if not allow_random_init:
                    raise RuntimeError(
                        f"CRITICAL: Failed to load pre-trained weights for '{model_source}': {err}.\n"
                        f"An active internet connection or local pre-trained cache is required for valid RoBERTa fine-tuning.\n"
                        f"If you intentionally want an uninitialized model for offline synthetic testing, set allow_random_init=True."
                    ) from err
                import sys
                print(
                    f"[WARNING] Could not load pre-trained weights from '{model_source}' ({err}). "
                    f"Falling back to randomly initialized architecture for offline tests.",
                    file=sys.stderr,
                )
                roberta_cfg = RobertaConfig(
                    hidden_size=self.custom_config.hidden_size,
                    max_position_embeddings=514,
                    type_vocab_size=1,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                )
                self.roberta = RobertaModel(roberta_cfg)

        self.custom_config.hidden_size = self.roberta.config.hidden_size

        self.layout_embeddings = LayoutEmbedding(
            hidden_size=self.roberta.config.hidden_size,
            max_2d_position_embeddings=self.custom_config.max_2d_position_embeddings,
            dropout=self.custom_config.classifier_dropout,
        )

        self.dropout = nn.Dropout(self.custom_config.classifier_dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        # Initialize weights of newly added layout and classifier heads
        self._init_weights(self.classifier)
        self.layout_embeddings.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        init_range = getattr(self.roberta.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=init_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass combining word token embeddings and 2D spatial layout embeddings.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            bbox: Tensor of shape (batch_size, seq_len, 4) with values in [0, 1000]
            attention_mask: Tensor of shape (batch_size, seq_len)
            token_type_ids: Optional token type ids
            labels: Optional BIO label ids of shape (batch_size, seq_len), with -100 for ignored tokens
            class_weights: Optional loss weights for entity balancing

        Returns:
            (loss, logits) if labels is provided, otherwise logits.
        """
        # 1. Compute word token embeddings from RoBERTa's embedding table
        words_embeddings = self.roberta.embeddings.word_embeddings(input_ids)

        # 2. Compute 2D spatial bounding box embeddings
        layout_embeddings = self.layout_embeddings(bbox)

        # 3. Fuse word embeddings and spatial embeddings
        fused_inputs_embeds = words_embeddings + layout_embeddings

        # 4. Generate proper position_ids from input_ids to account for padding tokens
        position_ids = None
        if hasattr(self.roberta.embeddings, "create_position_ids_from_input_ids"):
            position_ids = self.roberta.embeddings.create_position_ids_from_input_ids(
                input_ids, self.roberta.embeddings.padding_idx
            )
            max_pos = getattr(self.roberta.config, "max_position_embeddings", 514) - 1
            position_ids = position_ids.clamp(0, max_pos)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 5. Pass fused embeddings through RoBERTa encoder
        outputs = self.roberta(
            inputs_embeds=fused_inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits

    def save_pretrained(self, save_directory: Path | str) -> None:
        """Save model weights and configuration."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = save_path / "reconia_roberta_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.custom_config.to_dict(), f, indent=2)
        (save_path / "backbone_config.json").write_text(
            self.roberta.config.to_json_string(), encoding="utf-8"
        )

        weights_path = save_path / "model_weights.pt"
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: Path | str,
        device: str | torch.device = "cpu",
    ) -> RoBERTaWithLayoutForTokenClassification:
        """Load model from saved directory."""
        save_path = Path(save_directory)
        config_path = save_path / "reconia_roberta_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = RoBERTaWithLayoutConfig.from_dict(config_dict)
        backbone_path = save_path / "backbone_config.json"
        backbone = (
            RobertaModel(RobertaConfig.from_dict(json.loads(backbone_path.read_text(encoding="utf-8"))))
            if backbone_path.exists() else None
        )
        model = cls(config=config, roberta_model=backbone)

        weights_path = save_path / "model_weights.pt"
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        return model
