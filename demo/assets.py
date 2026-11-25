"""Sources of the synchronized Kaggle demo assets."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("roberta", "bros", "pick", "layoutlmv3", "donut", "qwen_vl")
DATASETS = ("sroie", "cord_v2", "wildreceipt")
NAMES = dict(zip(MODELS, ("RoBERTa", "BROS", "PICK", "LayoutLMv3", "Donut", "Qwen2.5-VL")))
DATA = Path(__file__).with_name("data")
MANIFEST = DATA / "artifacts.json"


def model_assets(model):
    return ROOT / "demo" / "model_assets" / model


def checkpoint(dataset, model):
    base = Path(os.environ.get("RECONIA_CHECKPOINT_DIR", ROOT / "checkpoints"))
    return base / dataset / model / "best_weights.pt"


def labels(dataset):
    return json.loads((DATA / "labels" / f"{dataset}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@lru_cache(maxsize=18)
def assert_checkpoint(dataset, model):
    info = manifest()["models"][model]["datasets"][dataset]
    path = checkpoint(dataset, model)
    if not path.is_file() or path.stat().st_size != info["checkpoint_bytes"]:
        raise ValueError("Checkpoint thiếu hoặc chưa được đồng bộ.")
    if file_hash(path) != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint khác với phiên Kaggle đang hiển thị trong demo.")
    return path


@lru_cache(maxsize=18)
def samples(dataset, model="layoutlmv3"):
    path = DATA / "samples.json"
    return json.loads(path.read_text(encoding="utf-8")).get(dataset, {}).get(model, {})


def sample_image(dataset, row):
    base = ROOT / "demo"
    path = (base / row["demo_image"]).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError("Thiếu ảnh mẫu demo.")
    return path
