import json
from pathlib import Path

from datasets import Dataset, Image as ImageFeature, load_from_disk

from utils.common import get_repo_dir, get_data_dir, normalize_box, export_labels


def build_gt(
    item: dict | list | str,
    prefix: str = "",
    gt: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    if gt is None:
        gt = {}

    if isinstance(item, dict):
        for key, value in item.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            build_gt(value, new_prefix, gt)
    elif isinstance(item, list):
        for value in item:
            build_gt(value, prefix, gt)
    elif item:
        gt.setdefault(prefix, []).append(item)

    return gt


def process_sample(raw_sample: dict, split_name: str) -> dict:
    annotation = json.loads(raw_sample["ground_truth"])
    sample_id = str(annotation["meta"]["image_id"])
    image_size = annotation["meta"]["image_size"]
    width, height = image_size["width"], image_size["height"]

    ocr: list[dict] = []
    for line in annotation["valid_line"]:
        category = line["category"]
        is_first = True

        for word in line["words"]:
            text = word["text"].strip()
            if not text:
                continue

            quad = [word["quad"][coord] for coord in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")]
            box = normalize_box(quad, width, height)

            is_key = word["is_key"]
            if is_key == 1 or not category:
                label = "O"
                is_first = True
            elif is_first:
                label = f"B-{category}"
                is_first = False
            else:
                label = f"I-{category}"

            ocr.append({
                "text": text,
                "box": box,
                "label": label,
            })

    image_path = f"images/{split_name}/{sample_id}.png"
    gt = build_gt(annotation["gt_parse"])

    return {
        "id": sample_id,
        "image_path": image_path,
        "ocr": ocr,
        "gt": gt,
    }


def export_split(split_dataset: Dataset, split_name: str, out_file: Path) -> set[str]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    labels: set[str] = set()

    with open(out_file, "w", encoding="utf-8") as file:
        for raw_sample in split_dataset:
            sample = process_sample(raw_sample, split_name)
            labels.update(item["label"] for item in sample["ocr"])
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return labels


def prepare_cord_v2() -> None:
    cache_dir = get_repo_dir() / ".cache" / "cord_v2"
    cord_v2_dir = get_data_dir("cord_v2")

    dataset = load_from_disk(cord_v2_dir)
    dataset = dataset.cast_column("image", ImageFeature(decode=False))

    train_labels = export_split(dataset["train"], "train", cache_dir / "train.jsonl")
    val_labels = export_split(dataset["validation"], "val", cache_dir / "val.jsonl")
    test_labels = export_split(dataset["test"], "test", cache_dir / "test.jsonl")

    export_labels(train_labels | val_labels | test_labels, "cord_v2")


if __name__ == "__main__":
    prepare_cord_v2()