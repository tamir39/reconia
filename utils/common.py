import json
import logging
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DATASETS = ("sroie", "cord_v2", "wildreceipt")
MODELS = ("roberta", "bros", "pick", "layoutlmv3", "donut", "qwen_vl")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_repo_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_dir(dataset_name: str) -> Path:
    configured = os.environ.get("RECONIA_DATA_DIR")
    roots = [Path(configured)] if configured else [
        get_repo_dir() / "data" / "raw",
        Path("/kaggle/input/datasets/vhngthnh/reconia-data"),
        Path("/kaggle/input/reconia-data"),
        Path("/kaggle/input/private-dataset"),
        get_repo_dir(),
    ]
    for root in roots:
        if (root / dataset_name).is_dir():
            return root / dataset_name
    raise FileNotFoundError("Set RECONIA_DATA_DIR to the directory containing the three raw datasets")


def get_cache_dir(dataset_name: str) -> Path:
    return Path(os.environ.get("RECONIA_PREPARED_DIR", get_repo_dir() / ".cache")) / dataset_name


def ensure_training_destination(repo_dir: Path, dataset: str, model: str, resume: bool) -> None:
    """Keep a new experiment from overwriting the submitted checkpoints/results."""
    existing = [repo_dir / "checkpoints" / dataset / model / name
                for name in ("best_weights.pt", "last_checkpoint.pt")]
    if not resume and any(path.exists() for path in existing):
        raise FileExistsError("Checkpoint already exists. Use a separate checkout for a new run, or --resume to continue it.")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_box(quad: list[int], width: int, height: int) -> list[int]:
    x_coords = quad[0::2]
    y_coords = quad[1::2]

    x_min = max(0, min(1000, int(min(x_coords) / width * 1000)))
    y_min = max(0, min(1000, int(min(y_coords) / height * 1000)))
    x_max = max(0, min(1000, int(max(x_coords) / width * 1000)))
    y_max = max(0, min(1000, int(max(y_coords) / height * 1000)))

    if x_max <= x_min:
        x_max = min(1000, x_min + 1)
        x_min = min(x_min, x_max - 1)

    if y_max <= y_min:
        y_max = min(1000, y_min + 1)
        y_min = min(y_min, y_max - 1)

    return [x_min, y_min, x_max, y_max]


def export_labels(labels: set[str], dataset_name: str) -> None:
    checkpoint_dir = get_repo_dir() / "checkpoints" / dataset_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ordered_labels = ["O"] + sorted(lbl for lbl in labels if lbl != "O")
    with open(checkpoint_dir / "labels.json", "w", encoding="utf-8") as file:
        json.dump(ordered_labels, file, indent=2, ensure_ascii=False)


def load_labels(dataset_name: str) -> list[str]:
    labels_file = get_repo_dir() / "checkpoints" / dataset_name / "labels.json"
    with open(labels_file, "r", encoding="utf-8") as file:
        return json.load(file)


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logger(logger)

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
