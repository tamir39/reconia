"""Utility script to download the CORD-v2 dataset from Hugging Face.

Usage:
    python download_cord.py [--dest PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.data.paths import WORKSPACE_ROOT


def download_cord(dest: Path) -> None:
    print(f"Downloading naver-clova-ix/cord-v2 from Hugging Face...")
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2")
    dest.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dest))
    print(f"Successfully saved CORD-v2 to: {dest.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CORD-v2 dataset from Hugging Face.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=WORKSPACE_ROOT / "cord",
        help="Destination directory (default: sibling 'cord' directory under workspace root).",
    )
    args = parser.parse_args()
    download_cord(args.dest)