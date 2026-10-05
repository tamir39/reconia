from pathlib import Path

import matplotlib.pyplot as plt


def save_train_plots(history: list[dict], out_dir: Path | str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    # 1. Loss Curve
    plt.figure()
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "loss.png")
    plt.close()

    # 2. Performance Metrics Curve
    plt.figure()
    plt.plot(epochs, [h["precision"] * 100 for h in history], label="Precision")
    plt.plot(epochs, [h["recall"] * 100 for h in history], label="Recall")
    plt.plot(epochs, [h["f1"] * 100 for h in history], label="F1")
    plt.title("Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "metrics.png")
    plt.close()


def save_comparison_plot(records: list[dict], out_dir: Path | str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models = [r["model"] for r in records]
    scores = [r["f1"] * 100 for r in records]

    plt.figure()
    bars = plt.bar(models, scores, width=0.5)
    plt.bar_label(bars, fmt="%.2f%%", padding=3)
    plt.title("Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 (%)")
    plt.margins(y=0.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path / "comparison.png")
    plt.close()


