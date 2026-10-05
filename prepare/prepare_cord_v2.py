import json
import os
from pathlib import Path

from datasets import Dataset, Image as ImageFeature, load_from_disk

from tools.common import get_repo_dir, resolve_data_dir, normalize_box


def flatten_dict(data: dict, prefix: str = "") -> dict[str, list[str]]:
    flattened = {}
    for key, val in data.items():
        field = f"{prefix}.{key}" if prefix else key

        if isinstance(val, dict):
            flattened.update(flatten_dict(val, prefix=field))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    sub_flattened = flatten_dict(item)
                    for sub_key, sub_val in sub_flattened.items():
                        sub_field = f"{field}.{sub_key}"
                        if isinstance(sub_val, list):
                            flattened.setdefault(sub_field, []).extend(
                                str(x) for x in sub_val if x is not None
                            )
                        elif sub_val is not None:
                            flattened.setdefault(sub_field, []).append(str(sub_val))
                elif isinstance(item, list):
                    for sub_item in item:
                        if sub_item is not None:
                            flattened.setdefault(field, []).append(str(sub_item))
                elif item is not None:
                    flattened.setdefault(field, []).append(str(item))
        else:
            if val is not None:
                flattened.setdefault(field, []).append(str(val))

    return flattened


def process_sample(sample_id: str, raw_sample: dict, image_dest_dir: Path) -> dict:
    gt_data = (
        json.loads(raw_sample["ground_truth"])
        if isinstance(raw_sample["ground_truth"], str)
        else raw_sample["ground_truth"]
    )
    meta = gt_data.get("meta", {})
    image_size = meta.get("image_size", {})
    width = image_size.get("width")
    height = image_size.get("height")

    image_path = image_dest_dir / f"{sample_id}.png"
    if not image_path.exists() and os.access(image_dest_dir, os.W_OK):
        image_path.write_bytes(raw_sample["image"]["bytes"])

    ocr_items = []
    for line in gt_data.get("valid_line", []):
        category = line.get("category")
        is_first = True

        for word in line.get("words", []):
            text = word.get("text", "").strip()
            if not text:
                continue

            quad_dict = word.get("quad", {})
            quad = [
                int(quad_dict.get("x1", 0)),
                int(quad_dict.get("y1", 0)),
                int(quad_dict.get("x2", 0)),
                int(quad_dict.get("y2", 0)),
                int(quad_dict.get("x3", 0)),
                int(quad_dict.get("y3", 0)),
                int(quad_dict.get("x4", 0)),
                int(quad_dict.get("y4", 0)),
            ]
            box = normalize_box(quad, width, height)

            if not category:
                label = "O"
            elif is_first:
                label = f"B-{category}"
                is_first = False
            else:
                label = f"I-{category}"

            ocr_items.append({
                "text": text,
                "quad": quad,
                "box": box,
                "label": label,
            })

    raw_gt = gt_data.get("gt_parse", {})
    gt = flatten_dict(raw_gt)

    return {
        "id": sample_id,
        "image_path": f"images/{image_dest_dir.name}/{sample_id}.png",
        "ocr": ocr_items,
        "gt": gt,
    }


def export_jsonl(dataset_split: Dataset, image_dest_dir: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not image_dest_dir.exists() and os.access(image_dest_dir.parent, os.W_OK):
        image_dest_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        for idx, raw_sample in enumerate(dataset_split):
            sample_id = str(idx)
            sample = process_sample(sample_id, raw_sample, image_dest_dir)
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_cord_v2() -> None:
    cache_dir = get_repo_dir() / ".cache" / "cord_v2"
    cord_v2_dir = resolve_data_dir("cord_v2")
    image_dir = cord_v2_dir / "images"

    if not cord_v2_dir.exists():
        from datasets import load_dataset
        dataset = load_dataset("naver-clova-ix/cord-v2")
        dataset.save_to_disk(str(cord_v2_dir))

    dataset = load_from_disk(str(cord_v2_dir))
    dataset = dataset.cast_column("image", ImageFeature(decode=False))

    # Original CORD-v2 splits: train (800), validation (100), test (100)
    export_jsonl(dataset["train"], image_dir / "train", cache_dir / "train.jsonl")
    export_jsonl(dataset["validation"], image_dir / "val", cache_dir / "val.jsonl")
    export_jsonl(dataset["test"], image_dir / "test", cache_dir / "test.jsonl")


if __name__ == "__main__":
    prepare_cord_v2()