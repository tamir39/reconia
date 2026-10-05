import argparse
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "roberta": "roberta-base",
    "bros": "clovaai/bros-base-uncased",
    "layoutlmv3": "microsoft/layoutlmv3-base",
    "donut": "naver-clova-ix/donut-base",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}

IGNORED_PATTERNS = [
    "*.msgpack",
    "*.h5",
    "*.ot",
    "*.onnx",
    "*.tflite",
    "*.rust_model.ot",
]


def download_model(model_name: str, weights_dir: Path) -> None:
    dest_dir = weights_dir / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODELS[model_name],
        local_dir=str(dest_dir),
        ignore_patterns=IGNORED_PATTERNS,
    )

    shutil.rmtree(dest_dir / ".cache", ignore_errors=True)


def download_weights(model_name: str = "all") -> None:
    weights_dir = Path(__file__).resolve().parents[1] / "weights"

    if model_name == "all":
        for name in MODELS:
            download_model(name, weights_dir)
    else:
        download_model(model_name, weights_dir)


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
