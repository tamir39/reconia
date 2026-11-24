from pathlib import Path

import matplotlib.pyplot as plt


def save_train_plots(history: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [log["epoch"] for log in history]

    plt.figure()
    plt.plot(epochs, [log["train_loss"] for log in history], label="Train Loss")
    plt.plot(epochs, [log["val_loss"] for log in history], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, [log["precision"] * 100 for log in history], label="Precision")
    plt.plot(epochs, [log["recall"] * 100 for log in history], label="Recall")
    plt.plot(epochs, [log["f1"] * 100 for log in history], label="F1")
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png")
    plt.close()


def save_comparison_plot(results: list[dict], dataset_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [item["model"] for item in results]
    scores = [item["f1"] * 100 for item in results]
    max_score = max(scores)
    colors = ["tab:orange" if score == max_score else "tab:blue" for score in scores]

    plt.figure()
    bars = plt.bar(models, scores, width=0.5, color=colors)
    plt.bar_label(bars, fmt="%.2f%%", padding=3)
    plt.title(f"Model Comparison of {dataset_name}")
    plt.xlabel("Model")
    plt.ylabel("F1 (%)")
    plt.margins(y=0.1)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png")
    plt.close()