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


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


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
