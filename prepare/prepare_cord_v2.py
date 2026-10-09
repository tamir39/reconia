import json
from pathlib import Path

from datasets import Dataset, Image as ImageFeature, load_from_disk

from tools.common import get_repo_dir, resolve_data_dir, normalize_box


def flatten_dict(
    data: dict | list | str,
    prefix: str = "",
    flattened: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    if flattened is None:
        flattened = {}

    if isinstance(data, dict):
        for key, val in data.items():
            sub_prefix = f"{prefix}.{key}" if prefix else key
            flatten_dict(val, prefix=sub_prefix, flattened=flattened)
    elif isinstance(data, list):
        for item in data:
            flatten_dict(item, prefix=prefix, flattened=flattened)
    elif data is not None:
        flattened.setdefault(prefix, []).append(str(data))

    return flattened


def process_sample(sample_id: str, raw_sample: dict, split_name: str) -> dict:
    gt_data = (
        json.loads(raw_sample["ground_truth"])
        if isinstance(raw_sample["ground_truth"], str)
        else raw_sample["ground_truth"]
    )
    meta = gt_data.get("meta", {})
    image_size = meta.get("image_size", {})
    width = image_size.get("width")
    height = image_size.get("height")

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

            is_key = word.get("is_key", 0)
            if is_key == 1 or not category:
                label = "O"
                is_first = True
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
        "image_path": f"images/{split_name}/{sample_id}.png",
        "ocr": ocr_items,
        "gt": gt,
    }


def export_jsonl(dataset_split: Dataset, split_name: str, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        for idx, raw_sample in enumerate(dataset_split):
            sample_id = str(idx)
            sample = process_sample(sample_id, raw_sample, split_name)
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_cord_v2() -> None:
    cache_dir = get_repo_dir() / ".cache" / "cord_v2"
    cord_v2_dir = resolve_data_dir("cord_v2")

    dataset = load_from_disk(str(cord_v2_dir))
    dataset = dataset.cast_column("image", ImageFeature(decode=False))

    # Original CORD-v2 splits: train (800), validation (100), test (100)
    export_jsonl(dataset["train"], "train", cache_dir / "train.jsonl")
    export_jsonl(dataset["validation"], "val", cache_dir / "val.jsonl")
    export_jsonl(dataset["test"], "test", cache_dir / "test.jsonl")


if __name__ == "__main__":
    prepare_cord_v2()