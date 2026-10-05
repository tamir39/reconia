import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(dataset_name: str) -> Path:
    repo_dir = Path(__file__).resolve().parents[1]
    candidates = [
        repo_dir.parent / dataset_name,
        Path("/kaggle/input/reconia-data") / dataset_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return repo_dir.parent / dataset_name


def normalize_box(quad: list[int], width: int, height: int) -> list[int]:
    x_coords = [quad[0], quad[2], quad[4], quad[6]]
    y_coords = [quad[1], quad[3], quad[5], quad[7]]

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


def setup_logger(out_dir: Path | str, name: str = "train") -> logging.Logger:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logger(logger)

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(out_path / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
