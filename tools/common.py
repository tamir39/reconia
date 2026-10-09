import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*nested tensor.*")

import numpy as np
import torch
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_repo_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_dir(dataset_name: str) -> Path:
    repo_dir = get_repo_dir()
    candidates = [
        repo_dir.parent / dataset_name,
        Path("/kaggle/input/datasets/vhngthnh/reconia-data") / dataset_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return repo_dir.parent / dataset_name


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+([,.:;!?])", r"\1", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_box(quad: list[int], width: int, height: int) -> list[int]:
    safe_width = max(width, 1)
    safe_height = max(height, 1)
    if len(quad) == 4:
        x_coords = [quad[0], quad[2]]
        y_coords = [quad[1], quad[3]]
    else:
        x_coords = [quad[0], quad[2], quad[4], quad[6]]
        y_coords = [quad[1], quad[3], quad[5], quad[7]]

    x_min = max(0, min(1000, int(min(x_coords) / safe_width * 1000)))
    y_min = max(0, min(1000, int(min(y_coords) / safe_height * 1000)))
    x_max = max(0, min(1000, int(max(x_coords) / safe_width * 1000)))
    y_max = max(0, min(1000, int(max(y_coords) / safe_height * 1000)))

    if x_max <= x_min:
        x_max = min(1000, x_min + 1)
        x_min = min(x_min, x_max - 1)

    if y_max <= y_min:
        y_max = min(1000, y_min + 1)
        y_min = min(y_min, y_max - 1)

    return [x_min, y_min, x_max, y_max]


def get_labels(cache_dir: Path) -> list[str]:
    labels_file = cache_dir / "labels.json"
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as file:
            return json.load(file)

    split_path = cache_dir / "train.jsonl"
    with open(split_path, "r", encoding="utf-8") as file:
        return sorted({
            item["label"]
            for line in file
            for item in json.loads(line).get("ocr", [])
            if item.get("label")
        })


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def setup_logger(out_dir: Path, name: str = "train") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
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