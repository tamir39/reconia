import difflib
import json
import random
import re
from pathlib import Path

from PIL import Image

from tools.common import get_repo_dir, resolve_data_dir, get_labels, normalize_text, normalize_box


def compute_token_quad(quad: list[int], start: int, end: int, total_len: int) -> list[int]:
    if total_len <= 0:
        return quad

    start_ratio = start / total_len
    end_ratio = end / total_len
    x1, y1, x2, y2, x3, y3, x4, y4 = quad

    return [
        int(round(x1 + start_ratio * (x2 - x1))),
        int(round(y1 + start_ratio * (y2 - y1))),
        int(round(x1 + end_ratio * (x2 - x1))),
        int(round(y1 + end_ratio * (y2 - y1))),
        int(round(x4 + end_ratio * (x3 - x4))),
        int(round(y4 + end_ratio * (y3 - y4))),
        int(round(x4 + start_ratio * (x3 - x4))),
        int(round(y4 + start_ratio * (y3 - y4))),
    ]


def parse_ocr_items(box_path: Path, width: int, height: int) -> list[dict]:
    ocr_items: list[dict] = []

    with open(box_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split(",", 8)
            if len(parts) != 9:
                continue

            quad = [int(coord) for coord in parts[:8]]
            line_text = parts[8].strip()
            total_len = len(line_text)

            for match in re.finditer(r"\S+", line_text):
                token_quad = compute_token_quad(quad, match.start(), match.end(), total_len)
                ocr_items.append({
                    "text": match.group(),
                    "quad": token_quad,
                    "box": normalize_box(token_quad, width, height),
                    "label": "O",
                })

    return ocr_items


def find_best_span(
    token_texts: list[str],
    target: str,
    is_reversed: bool = False,
) -> tuple[int, int] | None:
    if not target:
        return None

    total = len(token_texts)
    target_len = len(target)
    indices = range(total - 1, -1, -1) if is_reversed else range(total)

    for start_idx in indices:
        accum = ""
        for end_idx in range(start_idx, min(total, start_idx + 80)):
            accum += token_texts[end_idx]
            if accum == target:
                return start_idx, end_idx + 1
            if len(accum) >= target_len:
                break

    best_span = None
    best_score = 0.90
    min_len = int(target_len * 0.85)
    max_len = int(target_len * 1.15)

    for start_idx in indices:
        accum = ""
        for end_idx in range(start_idx, min(total, start_idx + 80)):
            accum += token_texts[end_idx]
            accum_len = len(accum)
            if accum_len >= min_len:
                score = difflib.SequenceMatcher(None, accum, target).ratio()
                if score > best_score:
                    best_score = score
                    best_span = (start_idx, end_idx + 1)
            if accum_len > max_len:
                break

    return best_span


def assign_labels(ocr_items: list[dict], raw_ground_truth: dict[str, str]) -> None:
    token_texts = [normalize_text(item["text"]) for item in ocr_items]

    for field in ("company", "date", "address", "total"):
        target = normalize_text(raw_ground_truth.get(field, ""))
        span = find_best_span(token_texts, target, is_reversed=(field == "total"))
        if not span:
            continue

        ocr_items[span[0]]["label"] = f"B-{field}"
        for idx in range(span[0] + 1, span[1]):
            ocr_items[idx]["label"] = f"I-{field}"


def process_sample(sample_id: str, split_dir: Path) -> dict:
    image_path = split_dir / "img" / f"{sample_id}.jpg"
    with Image.open(image_path) as image:
        width, height = image.size

    key_path = split_dir / "key" / f"{sample_id}.txt"
    with open(key_path, "r", encoding="utf-8", errors="ignore") as file:
        raw_ground_truth = json.load(file)

    box_path = split_dir / "box" / f"{sample_id}.txt"
    ocr_items = parse_ocr_items(box_path, width, height)
    assign_labels(ocr_items, raw_ground_truth)

    ground_truth = {key: [val] for key, val in raw_ground_truth.items() if val}

    return {
        "id": sample_id,
        "image_path": f"{split_dir.name}/img/{sample_id}.jpg",
        "ocr": ocr_items,
        "gt": ground_truth,
    }


def export_jsonl(sample_ids: list[str], split_dir: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        for sample_id in sample_ids:
            sample = process_sample(sample_id, split_dir)
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_sroie() -> None:
    cache_dir = get_repo_dir() / ".cache" / "sroie"
    sroie_dir = resolve_data_dir("sroie")

    # Train: 626 samples
    train_dir = sroie_dir / "train"
    # Test: 347 samples
    test_dir = sroie_dir / "test"

    train_ids = sorted(file_path.stem for file_path in (train_dir / "img").glob("*.jpg"))
    test_ids = sorted(file_path.stem for file_path in (test_dir / "img").glob("*.jpg"))

    random.seed(42)
    random.shuffle(train_ids)

    # 80/20 train/validation split with fixed seed 42
    val_ratio = 0.2
    val_count = int(len(train_ids) * val_ratio)
    val_ids = train_ids[:val_count]
    train_ids = train_ids[val_count:]

    export_jsonl(train_ids, train_dir, cache_dir / "train.jsonl")
    export_jsonl(val_ids, train_dir, cache_dir / "val.jsonl")
    export_jsonl(test_ids, test_dir, cache_dir / "test.jsonl")

    labels = get_labels(cache_dir)
    with open(cache_dir / "labels.json", "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    prepare_sroie()