import difflib
import json
import random
import re
from pathlib import Path

from PIL import Image

from utils.common import get_repo_dir, get_cache_dir, get_data_dir, normalize_text, normalize_box, export_labels


def load_ocr(box_file: Path, width: int, height: int) -> list[dict]:
    ocr: list[dict] = []

    with open(box_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split(",", 8)
            if len(parts) != 9:
                continue

            text = parts[8].strip()
            if not text:
                continue

            text_len = len(text)
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])

            for match in re.finditer(r"[^\s:]+|:", text):
                start_ratio = match.start() / text_len
                end_ratio = match.end() / text_len
                quad = [
                    int(round(x1 + start_ratio * (x2 - x1))),
                    int(round(y1 + start_ratio * (y2 - y1))),
                    int(round(x1 + end_ratio * (x2 - x1))),
                    int(round(y1 + end_ratio * (y2 - y1))),
                    int(round(x4 + end_ratio * (x3 - x4))),
                    int(round(y4 + end_ratio * (y3 - y4))),
                    int(round(x4 + start_ratio * (x3 - x4))),
                    int(round(y4 + start_ratio * (y3 - y4))),
                ]
                ocr.append({
                    "text": match.group(),
                    "box": normalize_box(quad, width, height),
                    "label": "O",
                })

    return ocr


def match_target(
    texts: list[str],
    target: str,
) -> tuple[int, int] | None:
    if not target:
        return None

    num_texts = len(texts)
    target_clean = target.replace(" ", "")
    target_len = len(target_clean)

    for start in range(num_texts):
        current_text = ""
        for end in range(start, min(num_texts, start + 64)):
            current_text += texts[end]
            if current_text == target_clean:
                return start, end + 1
            if len(current_text) >= target_len:
                break

    best_match = None
    best_score = 0.85
    min_len = int(target_len * 0.80)
    max_len = int(target_len * 1.20)

    for start in range(num_texts):
        current_text = ""
        for end in range(start, min(num_texts, start + 64)):
            current_text += texts[end]
            if len(current_text) >= min_len:
                score = difflib.SequenceMatcher(None, current_text, target_clean).ratio()
                if score > best_score:
                    best_score = score
                    best_match = (start, end + 1)
            if len(current_text) > max_len:
                break

    return best_match


def assign_labels(ocr: list[dict], raw_gt: dict[str, str]) -> None:
    texts = [normalize_text(item["text"]) for item in ocr]

    for entity in ("company", "date", "address", "total"):
        target = normalize_text(raw_gt.get(entity))
        match = match_target(texts, target)
        if not match:
            continue

        start, end = match
        ocr[start]["label"] = f"B-{entity}"
        for idx in range(start + 1, end):
            ocr[idx]["label"] = f"I-{entity}"


def process_sample(sample_id: str, split_dir: Path) -> dict:
    image_file = split_dir / "img" / f"{sample_id}.jpg"
    with Image.open(image_file) as image:
        width, height = image.size

    key_file = split_dir / "key" / f"{sample_id}.txt"
    with open(key_file, "r", encoding="utf-8", errors="ignore") as file:
        raw_gt = json.load(file)

    box_file = split_dir / "box" / f"{sample_id}.txt"
    ocr = load_ocr(box_file, width, height)
    assign_labels(ocr, raw_gt)

    image_path = f"{split_dir.name}/img/{sample_id}.jpg"
    gt = {entity: [text] for entity, text in raw_gt.items() if text}

    return {
        "id": sample_id,
        "image_path": image_path,
        "ocr": ocr,
        "gt": gt,
    }


def export_split(sample_ids: list[str], split_dir: Path, out_file: Path) -> set[str]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    labels: set[str] = set()

    with open(out_file, "w", encoding="utf-8") as file:
        for sample_id in sample_ids:
            sample = process_sample(sample_id, split_dir)
            labels.update(item["label"] for item in sample["ocr"])
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return labels


def prepare_sroie() -> None:
    cache_dir = get_cache_dir("sroie")
    sroie_dir = get_data_dir("sroie")

    train_dir = sroie_dir / "train"
    test_dir = sroie_dir / "test"

    train_ids = sorted(path.stem for path in (train_dir / "img").glob("*.jpg"))
    test_ids = sorted(path.stem for path in (test_dir / "img").glob("*.jpg"))

    random.seed(42)
    random.shuffle(train_ids)

    val_count = int(len(train_ids) * 0.2)
    val_ids = train_ids[:val_count]
    train_ids = train_ids[val_count:]

    train_labels = export_split(train_ids, train_dir, cache_dir / "train.jsonl")
    val_labels = export_split(val_ids, train_dir, cache_dir / "val.jsonl")
    test_labels = export_split(test_ids, test_dir, cache_dir / "test.jsonl")

    export_labels(train_labels | val_labels | test_labels, "sroie")


if __name__ == "__main__":
    prepare_sroie()
