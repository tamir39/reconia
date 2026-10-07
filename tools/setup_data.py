import argparse
import shutil
from pathlib import Path

from datasets import Image as ImageFeature, load_dataset, load_from_disk

from tools.common import get_repo_dir


DATASETS = ("sroie", "cord_v2")


def organize_sroie(sroie_dir: Path) -> None:
    train_dest = sroie_dir / "train"
    test_dest = sroie_dir / "test"

    # Train: 626 samples
    train_img = sroie_dir / "0325updated.task1train(626p)"
    train_box = sroie_dir / "0325updated.task1train(626p)"
    train_key = sroie_dir / "0325updated.task2train(626p)"

    if train_img.exists() and train_box.exists() and train_key.exists():
        (train_dest / "img").mkdir(parents=True, exist_ok=True)
        (train_dest / "box").mkdir(parents=True, exist_ok=True)
        (train_dest / "key").mkdir(parents=True, exist_ok=True)

        for file_path in train_img.glob("*.jpg"):
            if "(" in file_path.stem:
                continue
            sample_id = file_path.stem
            shutil.copy2(file_path, train_dest / "img" / f"{sample_id}.jpg")
            shutil.copy2(train_box / f"{sample_id}.txt", train_dest / "box" / f"{sample_id}.txt")
            shutil.copy2(train_key / f"{sample_id}.txt", train_dest / "key" / f"{sample_id}.txt")

    # Test: 347 samples
    test_img = sroie_dir / "SROIE_test_images_task_3"
    test_box = sroie_dir / "text.task1&2-test（361p)"
    test_key = sroie_dir / "SROIE_test_gt_task_3"

    if test_img.exists() and test_box.exists() and test_key.exists():
        (test_dest / "img").mkdir(parents=True, exist_ok=True)
        (test_dest / "box").mkdir(parents=True, exist_ok=True)
        (test_dest / "key").mkdir(parents=True, exist_ok=True)

        for file_path in test_img.glob("*.jpg"):
            sample_id = file_path.stem
            shutil.copy2(file_path, test_dest / "img" / f"{sample_id}.jpg")
            shutil.copy2(test_box / f"{sample_id}.txt", test_dest / "box" / f"{sample_id}.txt")
            shutil.copy2(test_key / f"{sample_id}.txt", test_dest / "key" / f"{sample_id}.txt")


def download_cord_v2(cord_v2_dir: Path) -> None:
    if not cord_v2_dir.exists():
        dataset = load_dataset("naver-clova-ix/cord-v2")
        dataset.save_to_disk(str(cord_v2_dir))

    dataset = load_from_disk(str(cord_v2_dir))
    dataset = dataset.cast_column("image", ImageFeature(decode=False))

    # CORD-v2 splits: train (800), validation (100), test (100)
    splits = [
        ("train", cord_v2_dir / "images" / "train"),
        ("validation", cord_v2_dir / "images" / "val"),
        ("test", cord_v2_dir / "images" / "test"),
    ]
    for split_name, dest_dir in splits:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(dataset[split_name]):
            image_path = dest_dir / f"{idx}.png"
            if not image_path.exists():
                image_path.write_bytes(sample["image"]["bytes"])


def setup_data(dataset: str = "all") -> None:
    repo_dir = get_repo_dir()
    target_datasets = list(DATASETS) if dataset == "all" else [dataset]

    if "sroie" in target_datasets:
        organize_sroie(repo_dir.parent / "sroie")

    if "cord_v2" in target_datasets:
        download_cord_v2(repo_dir.parent / "cord_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *DATASETS],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_data(args.dataset)