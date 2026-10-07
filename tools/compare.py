import json
from pathlib import Path

from tools.common import get_repo_dir
from tools.plots import save_comparison_plot

DATASETS = ("sroie", "cord_v2", "wildreceipt")
MODELS = ("roberta", "bros", "pick", "layoutlmv3", "donut", "qwen2_5_vl")


def collect_dataset_results(dataset_dir: Path) -> list[dict]:
    return [
        {
            "model": model,
            "test_loss": float(data.get("test_loss", 0.0)),
            "precision": float(data.get("precision", 0.0)),
            "recall": float(data.get("recall", 0.0)),
            "f1": float(data.get("f1", 0.0)),
        }
        for model in MODELS
        if (metric_file := dataset_dir / model / "test_metrics.json").exists()
        and (data := json.loads(metric_file.read_text(encoding="utf-8")))
    ]


def save_comparison_md(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.md", "w", encoding="utf-8") as file:
        file.write("| Model | Test Loss | Precision (%) | Recall (%) | F1 (%) |\n")
        file.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in records:
            file.write(
                f"| {r['model']} | {r['test_loss']:.4f} | "
                f"{r['precision'] * 100:.2f} | {r['recall'] * 100:.2f} | "
                f"{r['f1'] * 100:.2f} |\n"
            )


def compare_results() -> None:
    results_dir = get_repo_dir() / "results"

    for dataset in DATASETS:
        dataset_dir = results_dir / dataset
        records = collect_dataset_results(dataset_dir)
        if records:
            save_comparison_md(records, dataset_dir)
            save_comparison_plot(records, dataset_dir)


if __name__ == "__main__":
    compare_results()