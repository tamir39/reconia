import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from utils.common import get_repo_dir

MODELS = {
    "roberta": "roberta-base",
    "bros": "jinho8345/bros-base-uncased",
    "layoutlmv3": "microsoft/layoutlmv3-base",
    "donut": "naver-clova-ix/donut-base",
    "qwen_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}

IGNORED_PATTERNS = [
    ".git*",
    "*.h5",
    "*.md",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "*.tflite",
]


def download_model(model_name: str, weights_dir: Path) -> None:
    out_dir = weights_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODELS[model_name],
        local_dir=str(out_dir),
        ignore_patterns=IGNORED_PATTERNS,
    )


def download_weights(model_name: str = "all") -> None:
    weights_dir = get_repo_dir() / "weights"
    download_models = list(MODELS) if model_name == "all" else [model_name]
    for name in download_models:
        download_model(name, weights_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODELS],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_weights(args.model)
