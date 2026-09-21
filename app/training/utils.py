"""Shared utility functions and logging helpers for training pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


class DualLogger:
    """Duplicate stdout stream to both terminal and a persistent log file."""

    def __init__(self, log_path: Path) -> None:
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except UnicodeEncodeError:
            self.terminal.write(message.encode("ascii", errors="replace").decode("ascii"))
            self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self) -> int:
        return getattr(self.terminal, "fileno", lambda: 1)()

    def close(self) -> None:
        try:
            self.log_file.close()
        except Exception:
            pass

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.terminal, attr)


def str2bool(value: Any) -> bool:
    """Convert various string representations to boolean for CLI arguments."""
    if isinstance(value, bool):
        return value
    val_str = str(value).strip().lower()
    if val_str in ("yes", "true", "t", "y", "1"):
        return True
    if val_str in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")
