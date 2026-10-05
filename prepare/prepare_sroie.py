import json
import random
import re
from pathlib import Path

from PIL import Image

from tools.common import get_repo_dir, resolve_data_dir, normalize_box


def clean_alphanumeric(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(text)).lower()


def normalize_phrase(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    if not words:
        return ""
    return " " + " ".join(words) + " "


def match_box_label(text: str, gt: dict[str, str]) -> str | None:
    clean_line = clean_alphanumeric(text)
    if len(clean_line) <= 1:
        return None

    phrase_line = normalize_phrase(text)

    company = normalize_phrase(gt.get("company", ""))
    clean_company = clean_alphanumeric(gt.get("company", ""))
    if company and (company in phrase_line or phrase_line in company or clean_company == clean_line):
        return "company"

    date = clean_alphanumeric(gt.get("date", ""))
    if date and date in clean_line:
        return "date"

    address = normalize_phrase(gt.get("address", ""))
    if address and phrase_line in address:
        return "address"

    total = clean_alphanumeric(gt.get("total", ""))
    total_digits = re.sub(r"[^0-9]", "", gt.get("total", ""))
    line_digits = re.sub(r"[^0-9]", "", text)
    if (total and total in clean_line) or (total_digits and total_digits == line_digits):
        return "total"

    return None


def process_sample(sample_id: str, img_dir: Path, box_dir: Path, key_dir: Path) -> dict:
    image_path = img_dir / f"{sample_id}.jpg"
    with Image.open(image_path) as image:
        width, height = image.size

    key_path = key_dir / f"{sample_id}.txt"
    with open(key_path, "r", encoding="utf-8", errors="ignore") as file:
        gt = json.load(file)

    box_path = box_dir / f"{sample_id}.txt"
    ocr_items = []

    with open(box_path, "r", encoding="utf-8", errors="ignore") as file:
        prev_field = None
        for line in file:
            parts = line.strip().split(",", 8)
            if len(parts) != 9:
                continue

            text = parts[8].strip()
            quad = [int(coordinate) for coordinate in parts[:8]]
            box = normalize_box(quad, width, height)
            field = match_box_label(text, gt)

            if not field:
                label = "O"
                prev_field = None
            elif field != prev_field:
                label = f"B-{field}"
                prev_field = field
            else:
                label = f"I-{field}"

            ocr_items.append({
                "text": text,
                "quad": quad,
                "box": box,
                "label": label,
            })

    return {
        "id": sample_id,
        "image_path": f"{img_dir.parent.name}/img/{sample_id}.jpg",
        "ocr": ocr_items,
        "gt": gt,
    }


def export_jsonl(sample_ids: list[str], img_dir: Path, box_dir: Path, key_dir: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        for sample_id in sample_ids:
            sample = process_sample(sample_id, img_dir, box_dir, key_dir)
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_sroie() -> None:
    cache_dir = get_repo_dir() / ".cache" / "sroie"
    sroie_dir = resolve_data_dir("sroie")

    train_img = sroie_dir / "train" / "img"
    train_box = sroie_dir / "train" / "box"
    train_key = sroie_dir / "train" / "key"

    test_img = sroie_dir / "test" / "img"
    test_box = sroie_dir / "test" / "box"
    test_key = sroie_dir / "test" / "key"

    train_ids = sorted(file_path.stem for file_path in train_img.glob("*.jpg"))
    test_ids = sorted(file_path.stem for file_path in test_img.glob("*.jpg"))

    random.seed(42)
    random.shuffle(train_ids)

    # 80/20 train/validation split with fixed seed 42
    val_ratio = 0.2
    val_count = int(len(train_ids) * val_ratio)
    val_ids = train_ids[:val_count]
    train_ids = train_ids[val_count:]

    export_jsonl(train_ids, train_img, train_box, train_key, cache_dir / "train.jsonl")
    export_jsonl(val_ids, train_img, train_box, train_key, cache_dir / "val.jsonl")
    export_jsonl(test_ids, test_img, test_box, test_key, cache_dir / "test.jsonl")


if __name__ == "__main__":
    prepare_sroie()