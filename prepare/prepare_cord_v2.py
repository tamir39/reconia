import json
from pathlib import Path
from datasets import Image as ImageFeature, load_from_disk


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


def flatten_dict(data: dict, prefix: str = "") -> dict[str, list[str] | str]:
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
                        flattened.setdefault(sub_field, []).append(str(sub_val))
                elif item is not None:
                    flattened.setdefault(field, []).append(str(item))
        else:
            flattened[field] = val

    return flattened


def process_sample(sample_id: str, sample: dict, image_dest_dir: Path) -> dict:
    gt_data = (
        json.loads(sample["ground_truth"])
        if isinstance(sample["ground_truth"], str)
        else sample["ground_truth"]
    )
    meta = gt_data.get("meta", {})
    image_size = meta.get("image_size", {})
    width = image_size.get("width")
    height = image_size.get("height")

    image_path = image_dest_dir / f"{sample_id}.png"
    if not image_path.exists():
        image_bytes = sample["image"].get("bytes")
        if image_bytes:
            image_path.write_bytes(image_bytes)
        else:
            sample["image"].save(image_path)

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
        "ocr": ocr_items,
        "gt": gt,
    }


def export_jsonl(dataset_split, image_dest_dir: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_dest_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        for idx, sample in enumerate(dataset_split):
            sample_id = str(idx)
            sample_data = process_sample(sample_id, sample, image_dest_dir)
            file.write(json.dumps(sample_data, ensure_ascii=False) + "\n")


def prepare_cord_v2() -> None:
    cord_v2_dir = Path(__file__).resolve().parents[2] / "cord_v2"
    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "cord_v2"
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