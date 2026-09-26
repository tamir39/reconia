"""Pinned Lilt configuration and strict offline checkpoint reload."""

import torch
from transformers import LiltConfig, LiltForTokenClassification, LayoutLMv3TokenizerFast

from app.data.lilt_dataset import WINDOW_POLICY

BACKBONE_REPO = "SCUT-DLVCLab/lilt-roberta-en-base"
BACKBONE_REVISION = "96328faae824978e47bd0e8ea09f66b85efda5f5"


def load_lilt_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "lilt" or checkpoint.get("window_policy") != WINDOW_POLICY:
        raise ValueError("Unsupported Lilt checkpoint/window policy")
    config = LiltConfig.from_dict(checkpoint["config"])
    vocabulary = checkpoint["vocabulary"]
    if config.num_labels != len(vocabulary) or config.label2id != vocabulary:
        raise ValueError("Checkpoint label mapping does not match the model configuration")
    model = LiltForTokenClassification(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(path.parent / "tokenizer", local_files_only=True)
    return model.to(device).eval(), tokenizer, vocabulary
