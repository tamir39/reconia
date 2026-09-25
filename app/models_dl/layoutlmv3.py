"""Pinned LayoutLMv3 configuration and strict offline checkpoint reload."""

import torch
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification, AutoProcessor

from app.data.layoutlmv3_dataset import WINDOW_POLICY

BACKBONE_REPO = "microsoft/layoutlmv3-base"
BACKBONE_REVISION = "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"


def load_layoutlmv3_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "layoutlmv3" or checkpoint.get("window_policy") != WINDOW_POLICY:
        raise ValueError("Unsupported LayoutLMv3 checkpoint/window policy")
    config = LayoutLMv3Config.from_dict(checkpoint["config"])
    vocabulary = checkpoint["vocabulary"]
    if config.num_labels != len(vocabulary) or config.label2id != vocabulary:
        raise ValueError("Checkpoint label mapping does not match the model configuration")
    model = LayoutLMv3ForTokenClassification(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    processor = AutoProcessor.from_pretrained(path.parent / "processor", apply_ocr=False, local_files_only=True)
    return model.to(device).eval(), processor, vocabulary
