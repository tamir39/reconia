import argparse
import json
from pathlib import Path

from utils.common import DATASETS, MODELS, get_repo_dir
from utils.plots import save_comparison_plot


def collect_results(out_dir: Path) -> list[dict]:
    return [
        {
            "model": data["model"],
            "loss": data["loss"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1": data["f1"],
        }
        for model in MODELS
        if (metric_file := out_dir / model / "test_metrics.json").exists()
        and (data := json.loads(metric_file.read_text(encoding="utf-8")))
    ]


def save_comparison_md(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.md", "w", encoding="utf-8") as file:
        file.write("| Model | Loss | Precision | Recall | F1 |\n")
        file.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for item in results:
            file.write(
                f"| {item['model']} | {item['loss']:.4f} | "
                f"{item['precision']:.4f} | "
                f"{item['recall']:.4f} | "
                f"{item['f1']:.4f} |\n"
            )
        if any(item["model"] == "qwen_vl" for item in results):
            file.write("\nQwen: published result, not re-evaluated locally. Its evaluation generation budget uses "
                       "ground-truth answer length; see models/qwen_vl/README.md for the protocol note.\n")


def compare_results(source: Path | None = None, destination: Path | None = None) -> None:
    source = source or get_repo_dir() / "reports/benchmarks"
    destination = destination or get_repo_dir() / "output/comparison"
    for dataset_name in DATASETS:
        out_dir = destination / dataset_name
        results = collect_results(source / dataset_name)
        if results:
            save_comparison_md(results, out_dir)
            save_comparison_plot(results, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pinned Micro-F1 benchmarks; never replace the originals")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compare_results(args.source, args.output)
