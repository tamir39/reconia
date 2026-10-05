import json
import random
from pathlib import Path

from PIL import Image

from tools.common import get_repo_dir, resolve_data_dir, normalize_box


def load_class_map(class_list_file: Path) -> dict[int, str]:
    class_map = {}
    with open(class_list_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                class_id, class_name = parts
                class_map[int(class_id)] = class_name
    return class_map


def build_ground_truth(annotations: list[dict], class_map: dict[int, str]) -> dict[str, list[str]]:
    gt = {}
    for item in annotations:
        text = item.get("text", "").strip()
        if not text:
            continue

        label_name = class_map.get(item.get("label"), "")
        if not label_name.endswith("_value"):
            continue

        values = gt.setdefault(label_name, [])
        if label_name.startswith("Prod_") or text not in values:
            values.append(text)

    return gt


def process_sample(raw_sample: dict, wildreceipt_dir: Path, class_map: dict[int, str]) -> dict:
    image_path = raw_sample["file_name"]
    sample_id = Path(image_path).stem
    full_image_path = wildreceipt_dir / image_path

    # Read dimensions from image file to ensure correct scaling
    with Image.open(full_image_path) as image:
        width, height = image.size

    ocr_items = []
    for item in raw_sample.get("annotations", []):
        text = item.get("text", "").strip()
        if not text:
            continue

        quad = [int(round(coord)) for coord in item.get("box", [])]
        box = normalize_box(quad, width, height)
        raw_label = class_map.get(item.get("label"), "")
        if raw_label.endswith("_value"):
            label = f"B-{raw_label}"
        else:
            label = "O"

        ocr_items.append({
            "text": text,
            "quad": quad,
            "box": box,
            "label": label,
        })

    gt = build_ground_truth(raw_sample.get("annotations", []), class_map)

    return {
        "id": sample_id,
        "image_path": image_path,
        "ocr": ocr_items,
        "gt": gt,
    }


def export_jsonl(raw_samples: list[dict], wildreceipt_dir: Path, class_map: dict[int, str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        for raw_sample in raw_samples:
            sample = process_sample(raw_sample, wildreceipt_dir, class_map)
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_wildreceipt() -> None:
    cache_dir = get_repo_dir() / ".cache" / "wildreceipt"
    wildreceipt_dir = resolve_data_dir("wildreceipt")

    class_map = load_class_map(wildreceipt_dir / "class_list.txt")

    with open(wildreceipt_dir / "train.txt", "r", encoding="utf-8") as file:
        train_raw_samples = [json.loads(line) for line in file if line.strip()]

    with open(wildreceipt_dir / "test.txt", "r", encoding="utf-8") as file:
        test_raw_samples = [json.loads(line) for line in file if line.strip()]

    random.seed(42)
    random.shuffle(train_raw_samples)

    # 80/20 train/validation split with fixed seed 42
    val_ratio = 0.2
    val_count = int(len(train_raw_samples) * val_ratio)
    val_raw_samples = train_raw_samples[:val_count]
    train_raw_samples = train_raw_samples[val_count:]

    export_jsonl(train_raw_samples, wildreceipt_dir, class_map, cache_dir / "train.jsonl")
    export_jsonl(val_raw_samples, wildreceipt_dir, class_map, cache_dir / "val.jsonl")
    export_jsonl(test_raw_samples, wildreceipt_dir, class_map, cache_dir / "test.jsonl")


if __name__ == "__main__":
    prepare_wildreceipt()