import difflib
import json
import random
import re
from pathlib import Path

from PIL import Image

from tools.common import get_repo_dir, resolve_data_dir, normalize_box


def clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(text)).lower()


def compute_token_quad(quad: list[int], start: int, end: int, total_len: int) -> list[int]:
    if total_len <= 0:
        return quad

    ratio_start = start / total_len
    ratio_end = end / total_len
    x1, y1, x2, y2, x3, y3, x4, y4 = quad

    point1_x = int(round(x1 + ratio_start * (x2 - x1)))
    point1_y = int(round(y1 + ratio_start * (y2 - y1)))
    point2_x = int(round(x1 + ratio_end * (x2 - x1)))
    point2_y = int(round(y1 + ratio_end * (y2 - y1)))
    point3_x = int(round(x4 + ratio_end * (x3 - x4)))
    point3_y = int(round(y4 + ratio_end * (y3 - y4)))
    point4_x = int(round(x4 + ratio_start * (x3 - x4)))
    point4_y = int(round(y4 + ratio_start * (y3 - y4)))

    return [point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y]


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


def find_best_span(token_texts: list[str], target: str) -> tuple[int, int] | None:
    if not target:
        return None

    total = len(token_texts)
    target_len = len(target)

    for start_idx in range(total):
        if not token_texts[start_idx]:
            continue

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

    for start_idx in range(total):
        if not token_texts[start_idx]:
            continue

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


def assign_labels(ocr_items: list[dict], gt: dict[str, str]) -> None:
    token_texts = [clean_text(item["text"]) for item in ocr_items]

    for field in ("company", "date", "address", "total"):
        target = clean_text(gt.get(field, ""))
        span = find_best_span(token_texts, target)
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
        gt_raw = json.load(file)

    box_path = split_dir / "box" / f"{sample_id}.txt"
    ocr_items = parse_ocr_items(box_path, width, height)
    assign_labels(ocr_items, gt_raw)

    gt = {key: [val] for key, val in gt_raw.items() if val}

    return {
        "id": sample_id,
        "image_path": f"{split_dir.name}/img/{sample_id}.jpg",
        "ocr": ocr_items,
        "gt": gt,
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


if __name__ == "__main__":
    prepare_sroie()