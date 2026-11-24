import json
import random
from pathlib import Path

from PIL import Image

from utils.common import get_repo_dir, get_data_dir, normalize_box, export_labels


def load_class_map(class_list_file: Path) -> dict[int, str]:
    with open(class_list_file, "r", encoding="utf-8") as file:
        return {
            int(parts[0]): parts[1]
            for line in file
            if len(parts := line.split()) == 2
        }


def build_gt(annotations: list[dict], class_map: dict[int, str]) -> dict[str, list[str]]:
    gt: dict[str, list[str]] = {}
    for item in annotations:
        text = item["text"].strip()
        if not text:
            continue

        label_name = class_map[item["label"]]
        if not label_name.endswith("_value"):
            continue

        gt.setdefault(label_name, []).append(text)

    return gt


def process_sample(raw_sample: dict, wildreceipt_dir: Path, class_map: dict[int, str]) -> dict:
    image_path = raw_sample["file_name"]
    sample_id = Path(image_path).stem
    image_file = wildreceipt_dir / image_path

    with Image.open(image_file) as image:
        width, height = image.size

    ocr: list[dict] = []
    for item in raw_sample["annotations"]:
        text = item["text"].strip()
        if not text:
            continue

        quad = [int(round(coord)) for coord in item["box"]]
        box = normalize_box(quad, width, height)

        label_name = class_map[item["label"]]
        label = f"B-{label_name}" if label_name.endswith("_value") else "O"

        ocr.append({
            "text": text,
            "box": box,
            "label": label,
        })

    gt = build_gt(raw_sample["annotations"], class_map)

    return {
        "id": sample_id,
        "image_path": image_path,
        "ocr": ocr,
        "gt": gt,
    }


def export_split(
    raw_samples: list[dict],
    wildreceipt_dir: Path,
    out_file: Path,
    class_map: dict[int, str],
) -> set[str]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    labels: set[str] = set()

    with open(out_file, "w", encoding="utf-8") as file:
        for raw_sample in raw_samples:
            sample = process_sample(raw_sample, wildreceipt_dir, class_map)
            labels.update(item["label"] for item in sample["ocr"])
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return labels


def prepare_wildreceipt() -> None:
    cache_dir = get_repo_dir() / ".cache" / "wildreceipt"
    wildreceipt_dir = get_data_dir("wildreceipt")

    class_map = load_class_map(wildreceipt_dir / "class_list.txt")

    with open(wildreceipt_dir / "train.txt", "r", encoding="utf-8") as file:
        train_raw_samples = [json.loads(line) for line in file if line.strip()]

    with open(wildreceipt_dir / "test.txt", "r", encoding="utf-8") as file:
        test_raw_samples = [json.loads(line) for line in file if line.strip()]

    random.seed(42)
    random.shuffle(train_raw_samples)

    val_count = int(len(train_raw_samples) * 0.2)
    val_raw_samples = train_raw_samples[:val_count]
    train_raw_samples = train_raw_samples[val_count:]

    train_labels = export_split(train_raw_samples, wildreceipt_dir, cache_dir / "train.jsonl", class_map)
    val_labels = export_split(val_raw_samples, wildreceipt_dir, cache_dir / "val.jsonl", class_map)
    test_labels = export_split(test_raw_samples, wildreceipt_dir, cache_dir / "test.jsonl", class_map)

    export_labels(train_labels | val_labels | test_labels, "wildreceipt")


if __name__ == "__main__":
    prepare_wildreceipt()