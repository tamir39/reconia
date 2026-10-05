import json
import random
from pathlib import Path
from PIL import Image


def load_class_map(class_list_file: Path) -> dict[int, str]:
    class_map = {}
    with open(class_list_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                class_id, class_name = parts
                class_map[int(class_id)] = class_name
    return class_map


def normalize_box(quad: list[int], width: int, height: int) -> list[int]:
    x_coords = [quad[0], quad[2], quad[4], quad[6]]
    y_coords = [quad[1], quad[3], quad[5], quad[7]]

    x_min = max(0, min(1000, int(min(x_coords) / width * 1000)))
    y_min = max(0, min(1000, int(min(y_coords) / height * 1000)))
    x_max = max(0, min(1000, int(max(x_coords) / width * 1000)))
    y_max = max(0, min(1000, int(max(y_coords) / height * 1000)))

    if x_max <= x_min:
        x_max = min(1000, x_min + 1)
        if x_max <= x_min:
            x_min = 999

    if y_max <= y_min:
        y_max = min(1000, y_min + 1)
        if y_max <= y_min:
            y_min = 999

    return [x_min, y_min, x_max, y_max]


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
    sample_id = raw_sample["file_name"]
    image_path = wildreceipt_dir / sample_id

    # Read dimensions from image file to ensure correct scaling
    with Image.open(image_path) as image:
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
        "ocr": ocr_items,
        "gt": gt,
    }


def export_jsonl(samples: list[dict], wildreceipt_dir: Path, class_map: dict[int, str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        for sample in samples:
            sample_data = process_sample(sample, wildreceipt_dir, class_map)
            file.write(json.dumps(sample_data, ensure_ascii=False) + "\n")


def prepare_wildreceipt() -> None:
    wildreceipt_dir = Path(__file__).resolve().parents[2] / "wildreceipt"
    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "wildreceipt"

    class_map = load_class_map(wildreceipt_dir / "class_list.txt")

    train_samples = []
    with open(wildreceipt_dir / "train.txt", "r", encoding="utf-8") as file:
        for line in file:
            line_str = line.strip()
            if line_str:
                train_samples.append(json.loads(line_str))

    test_samples = []
    with open(wildreceipt_dir / "test.txt", "r", encoding="utf-8") as file:
        for line in file:
            line_str = line.strip()
            if line_str:
                test_samples.append(json.loads(line_str))

    random.seed(42)
    random.shuffle(train_samples)

    # 80/20 train/validation split with fixed seed 42
    val_ratio = 0.2
    val_count = int(len(train_samples) * val_ratio)
    val_samples = train_samples[:val_count]
    train_samples = train_samples[val_count:]

    export_jsonl(train_samples, wildreceipt_dir, class_map, cache_dir / "train.jsonl")
    export_jsonl(val_samples, wildreceipt_dir, class_map, cache_dir / "val.jsonl")
    export_jsonl(test_samples, wildreceipt_dir, class_map, cache_dir / "test.jsonl")


if __name__ == "__main__":
    prepare_wildreceipt()