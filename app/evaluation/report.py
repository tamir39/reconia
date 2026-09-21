"""Reporting and visualization module for deep learning invoice extraction.

Exports:
  - Detailed JSON reports (machine-readable, complete history, per-class breakdown).
  - CSV history tables (compatible with Excel, Pandas).
  - Markdown summary reports (human-readable, ready for thesis documentation).
  - High-resolution publication-quality PNG charts (Loss curves, F1 curves, per-class bar charts).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    # Use non-interactive backend suitable for headless and CLI execution
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_training_history_csv(csv_path: Path, history: list[dict[str, Any]]) -> None:
    """Save epoch-by-epoch training metrics to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "precision",
        "recall",
        "f1",
        "latency",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            clean_row = {
                "epoch": row.get("epoch", 1),
                "train_loss": row.get("train_loss", 0.0),
                "val_loss": row.get("val_loss", 0.0),
                "precision": row.get("precision", 0.0),
                "recall": row.get("recall", 0.0),
                "f1": row.get("f1", 0.0),
                "latency": row.get("latency", 0.0),
            }
            writer.writerow(clean_row)


def save_markdown_report(
    md_path: Path,
    summary_data: dict[str, Any],
    figure_paths: list[Path] | None = None,
) -> None:
    """Generate a clean, readable Markdown report of training and evaluation."""
    md_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = summary_data.get("model", "Model")
    dataset_name = summary_data.get("dataset", "Dataset").upper()
    best_epoch = summary_data.get("best_epoch", 1)
    train_loss = summary_data.get("train_loss", 0.0)
    val_loss = summary_data.get("val_loss", 0.0)
    precision = summary_data.get("precision", 0.0)
    recall = summary_data.get("recall", 0.0)
    f1 = summary_data.get("f1", 0.0)
    total_params = summary_data.get("total_params", 0)
    latency = summary_data.get("latency", 0.0)
    device = summary_data.get("device", "CPU")
    history = summary_data.get("history", [])
    per_type = summary_data.get("per_type", summary_data.get("entity_per_type", {}))
    metrics_val = summary_data.get("metrics_val")
    metrics_test = summary_data.get("metrics_test")

    lines: list[str] = [
        f"# Báo Cáo Thực Nghiệm: {model_name} trên Bộ Dữ Liệu {dataset_name}",
        "",
    ]
    sec_idx = 1
    lines.append(f"## {sec_idx}. Kết Quả Thực Nghiệm Tối Ưu")
    lines.append("")

    if metrics_val and metrics_test:
        v_prec = metrics_val.get("precision", 0.0) * 100
        v_rec = metrics_val.get("recall", 0.0) * 100
        v_f1 = metrics_val.get("f1", 0.0) * 100
        v_lat = metrics_val.get("latency", 0.0)

        t_prec = metrics_test.get("precision", 0.0) * 100
        t_rec = metrics_test.get("recall", 0.0) * 100
        t_f1 = metrics_test.get("f1", 0.0) * 100
        t_lat = metrics_test.get("latency", 0.0)

        lines.append("| Chỉ số | Tập Validation | Tập Test |")
        lines.append("|---|:---:|:---:|")
        lines.append(f"| **Precision** | {v_prec:.2f}% | **{t_prec:.2f}%** |")
        lines.append(f"| **Recall** | {v_rec:.2f}% | **{t_rec:.2f}%** |")
        lines.append(f"| **F1-Score** | {v_f1:.2f}% | **{t_f1:.2f}%** |")
        lines.append(f"| **Latency** | {v_lat:.2f} ms | {t_lat:.2f} ms |")
        lines.append("")
        lines.append(f"- **Mô hình**: `{model_name}` | **Thiết bị**: `{device}` | **Tham số**: {total_params:,} | **Epoch tối ưu**: {best_epoch}")
        lines.append(f"- **Train Loss**: {train_loss:.4f} | **Val Loss**: {val_loss:.4f}")
    else:
        lines.append("| Tiêu chí | Giá trị |")
        lines.append("|---|---|")
        lines.append(f"| **Mô hình** | `{model_name}` |")
        lines.append(f"| **Bộ dữ liệu** | `{dataset_name}` |")
        lines.append(f"| **Thiết bị** | `{device}` |")
        lines.append(f"| **Tham số** | {total_params:,} |")
        lines.append(f"| **Epoch tối ưu** | {best_epoch} |")
        lines.append(f"| **Train Loss** | {train_loss:.4f} |")
        lines.append(f"| **Val Loss** | {val_loss:.4f} |")
        lines.append(f"| **Precision** | **{precision * 100:.2f}%** |")
        lines.append(f"| **Recall** | **{recall * 100:.2f}%** |")
        lines.append(f"| **F1** | **{f1 * 100:.2f}%** |")
        lines.append(f"| **Latency (ms)** | **{latency:.2f}** |")
    lines.append("")

    # Per-type Entity Table (Section 2: Detailed breakdown right after summary)
    if per_type:
        sec_idx += 1
        lines.append(f"## {sec_idx}. Kết Quả Theo Từng Nhãn Thực Thể")
        lines.append("")
        lines.append("| Nhãn | Precision | Recall | F1 | Mẫu (Support) |")
        lines.append("|---|---|---|---|---|")
        for ent_name, stats in sorted(per_type.items()):
            p = stats.get("precision", 0.0) * 100
            r = stats.get("recall", 0.0) * 100
            ent_f1 = stats.get("f1", 0.0) * 100
            supp = stats.get("support", 0)
            lines.append(f"| `{ent_name}` | {p:.2f}% | {r:.2f}% | **{ent_f1:.2f}%** | {supp:,} |")
        lines.append(f"| **Tổng thể** | **{precision * 100:.2f}%** | **{recall * 100:.2f}%** | **{f1 * 100:.2f}%** | — |")
        lines.append("")

    # Embedded Figures (Section 3: Training dynamics visualization)
    if figure_paths:
        sec_idx += 1
        lines.append(f"## {sec_idx}. Biểu Đồ Huấn Luyện")
        lines.append("")
        for fig_p in figure_paths:
            rel_path = f"figures/{fig_p.name}" if fig_p.parent.name == "figures" else fig_p.name
            lines.append(f"![{fig_p.stem}]({rel_path})")
            lines.append("")

    # History Table (Section 4: Epoch history)
    if history:
        sec_idx += 1
        lines.append(f"## {sec_idx}. Lịch Sử Huấn Luyện")
        lines.append("")
        lines.append("| Epoch | Train Loss | Val Loss | Precision | Recall | F1 | Latency (ms) |")
        lines.append("|---|---|---|---|---|---|---|")
        for h in history:
            ep = h.get("epoch", 0)
            tl = h.get("train_loss", 0.0)
            vl = h.get("val_loss", 0.0)
            p = h.get("precision", 0.0) * 100
            r = h.get("recall", 0.0) * 100
            ef = h.get("f1", 0.0) * 100
            lat = h.get("latency", 0.0)
            lines.append(f"| {ep} | {tl:.4f} | {vl:.4f} | {p:.2f}% | {r:.2f}% | **{ef:.2f}%** | {lat:.1f} |")
        lines.append("")

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def plot_training_dashboard(
    figure_path: Path,
    history: list[dict[str, Any]],
    per_type_metrics: dict[str, dict[str, float]],
    model_name: str,
    dataset_name: str,
) -> Path | None:
    """Generate a clean 2-panel training and evaluation figure (Loss & Metrics)."""
    if plt is None:
        print("[Warning] matplotlib is not installed. Skipping chart generation.")
        return None

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return None

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    prec = [h.get("precision", 0.0) * 100 for h in history]
    rec = [h.get("recall", 0.0) * 100 for h in history]
    f1 = [h.get("f1", 0.0) * 100 for h in history]
    latencies = [h.get("latency", 0.0) for h in history]

    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    fig.suptitle(
        f"{model_name} on {dataset_name.upper()}  (Latency: {avg_lat:.1f} ms)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Loss Curves
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, "o-", color="#1f77b4", label="Train Loss", linewidth=2, markersize=5)
    ax1.plot(epochs, val_loss, "s--", color="#d62728", label="Val Loss", linewidth=2, markersize=5)
    max_loss = max(max(train_loss, default=1.0), max(val_loss, default=1.0))
    ax1.set_ylim(bottom=0, top=max_loss * 1.2 + 0.1)
    ax1.set_title("Hội Tụ Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Precision, Recall & F1 Curves
    ax2 = axes[1]
    ax2.plot(epochs, prec, "o-", color="#17becf", label="Precision (%)", linewidth=2, markersize=5)
    ax2.plot(epochs, rec, "s--", color="#ff7f0e", label="Recall (%)", linewidth=2, markersize=5)
    ax2.plot(epochs, f1, "^-", color="#2ca02c", label="F1 (%)", linewidth=2, markersize=5)
    ax2.set_ylim(0, 105)
    ax2.set_title("Chất Lượng Trích Xuất (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Tỷ lệ (%)", fontsize=10)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Ensure integer ticks for epochs
    if len(epochs) <= 20:
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(figure_path, format="png", bbox_inches="tight")
    plt.close(fig)
    return figure_path


def export_full_benchmark_report(
    output_dir: Path,
    model_name: str,
    dataset_name: str,
    history: list[dict[str, Any]],
    final_val_metrics: dict[str, Any],
    metadata: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Orchestrate saving JSON, CSV, Markdown, and Chart figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{model_name.lower()}_{dataset_name.lower()}"
    json_path = metrics_dir / f"{base_name}_results.json"
    csv_path = metrics_dir / f"{base_name}_history.csv"
    md_path = output_dir / f"{base_name}_report.md"
    fig_path = figures_dir / f"{base_name}_training_curves.png"

    # Determine best epoch
    best_epoch = metadata.get("best_epoch")
    if best_epoch is not None and history:
        best_entry = next((e for e in history if e.get("epoch") == best_epoch), {})
    else:
        best_entry = max(history, key=lambda x: x.get("f1", 0.0)) if history else {}
        best_epoch = best_entry.get("epoch", 1)

    val_precision = float(best_entry.get("precision", 0.0))
    val_recall = float(best_entry.get("recall", 0.0))
    val_f1 = float(best_entry.get("f1", 0.0))
    val_lat = float(best_entry.get("latency", 0.0))

    metrics_val = {
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
        "train_loss": float(best_entry.get("train_loss", 0.0)),
        "val_loss": float(best_entry.get("val_loss", 0.0)),
        "latency": val_lat,
    }

    metrics_test = None
    if test_metrics is not None:
        t_perf = test_metrics.get("perf", {})
        metrics_test = {
            "precision": float(test_metrics.get("precision", 0.0)),
            "recall": float(test_metrics.get("recall", 0.0)),
            "f1": float(test_metrics.get("f1", 0.0)),
            "latency": float(test_metrics.get("latency", t_perf.get("latency", 0.0))),
        }

    # Primary reported metrics: Prefer TEST set if available, fallback to VAL
    primary_metrics = metrics_test if metrics_test is not None else metrics_val
    split_used = "TEST" if metrics_test is not None else "VAL"

    # Per-type entity stats
    per_type = (
        final_val_metrics.get("per_type")
        or final_val_metrics.get("entity_metrics", {}).get("per_type")
        or {}
    )

    summary_data = {
        "model": model_name,
        "dataset": dataset_name,
        "best_epoch": best_epoch,
        "split": split_used,
        "train_loss": metrics_val["train_loss"],
        "val_loss": metrics_val["val_loss"],
        "precision": primary_metrics["precision"],
        "recall": primary_metrics["recall"],
        "f1": primary_metrics["f1"],
        "latency": primary_metrics["latency"],
        "total_params": metadata.get("total_params", 0),
        "device": metadata.get("device", "CPU"),
        "history": history,
        "per_type": per_type,
        "metadata": metadata,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
    }

    # 1. Save JSON
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_data, fh, indent=2)

    # 2. Save CSV
    save_training_history_csv(csv_path, history)

    # 3. Generate Chart Figure
    generated_fig = plot_training_dashboard(
        fig_path,
        history,
        per_type,
        model_name,
        dataset_name,
    )

    # 4. Save Markdown Report with embedded chart
    fig_list = [generated_fig] if generated_fig else []
    save_markdown_report(md_path, summary_data, figure_paths=fig_list)

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "figure": fig_path,
    }



def generate_benchmark_summary(reports_dir: Path) -> dict[str, Path | None]:
    """Scan all *_results.json in reports_dir, build comparative tables and charts.

    Exports:
      - reports/benchmark_summary.md (comprehensive cross-model comparison table)
      - reports/figures/benchmark_comparison.png (grouped bar charts of F1 and Latency)
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports_dir / "metrics"

    json_files = sorted(metrics_dir.glob("*_results.json")) if metrics_dir.is_dir() else []
    if not json_files:
        json_files = sorted(reports_dir.glob("*_results.json"))
    if not json_files:
        return {"markdown": None, "figure": None}

    records: list[dict[str, Any]] = []
    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                records.append(data)
        except Exception:
            continue

    if not records:
        return {"markdown": None, "figure": None}

    # Sort records by dataset, then model
    records.sort(key=lambda r: (r.get("dataset", ""), r.get("model", "")))

    # 1. Generate Markdown Comparison Table
    md_path = reports_dir / "benchmark_summary.md"
    md_lines = [
        "# Bảng So Sánh Tổng Hợp Các Mô Hình",
        "",
        "Tổng hợp kết quả thực nghiệm từ các file trong thư mục `reports/` (ưu tiên đánh giá trên tập **TEST** độc lập):",
        "",
        "## 1. Kết Quả Tổng Hợp",
        "",
        "| Mô hình | Bộ dữ liệu | Tập | Epoch | Precision (%) | Recall (%) | F1 (%) | Latency (ms) |",
        "|---|---|:---:|:---:|---|---|---|---|",
    ]

    for r in records:
        model = r.get("model", "Unknown")
        dataset = r.get("dataset", "Unknown").upper()
        best_ep = r.get("best_epoch", 1)
        split = r.get("split", "TEST" if r.get("metrics_test") else "VAL")
        prec = float(r.get("precision", 0.0)) * 100
        rec = float(r.get("recall", 0.0)) * 100
        f1_val = float(r.get("f1", 0.0)) * 100
        lat = r.get("latency", 0.0)

        md_lines.append(
            f"| **{model}** | `{dataset}` | `{split}` | {best_ep} | {prec:.2f}% | {rec:.2f}% | **{f1_val:.2f}%** | {lat:.2f} ms |"
        )

    fig_path = figures_dir / "benchmark_comparison.png"
    has_fig = False

    # 2. Generate Comparative Bar Chart if matplotlib is available (supports N models dynamically)
    if plt is not None and len(records) >= 1:
        try:
            labels = [f"{r.get('model', '')}\n({r.get('dataset', '').upper()} - {r.get('split', 'TEST' if r.get('metrics_test') else 'VAL')})" for r in records]
            f1s = [float(r.get("f1", 0.0)) * 100 for r in records]
            latencies = [r.get("latency", 0.0) for r in records]

            fig_width = max(11.0, len(labels) * 1.8)
            fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5.0), dpi=300)
            fig.suptitle("So Sánh Hiệu Năng Các Mô Hình (Đánh Giá Tập TEST)", fontsize=14, fontweight="bold", y=0.98)

            x = np.arange(len(labels))

            # Subplot 1: F1
            ax1 = axes[0]
            bars_f1 = ax1.bar(x, f1s, width=0.45, color="#2ca02c", edgecolor="black", linewidth=0.5)
            ax1.set_title("F1 (%)", fontsize=11, fontweight="bold")
            ax1.set_ylabel("F1 (%)", fontsize=10)
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, fontsize=9)
            ax1.set_ylim(0, 115)
            ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
            for bar in bars_f1:
                h = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.5,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="semibold",
                )

            # Subplot 2: Latency
            ax2 = axes[1]
            bars_lat = ax2.bar(x, latencies, width=0.45, color="#1f77b4", edgecolor="black", linewidth=0.5)
            ax2.set_title("Latency (ms)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Latency (ms)", fontsize=10)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=9)
            max_lat = max(latencies) if latencies else 1.0
            ax2.set_ylim(0, max_lat * 1.28 + 0.5)
            ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
            for bar in bars_lat:
                h = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (max_lat * 0.02 + 0.1),
                    f"{h:.1f} ms",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="semibold",
                )

            plt.tight_layout(rect=[0, 0.03, 1, 0.94])
            plt.savefig(fig_path, format="png", bbox_inches="tight")
            plt.close(fig)
            has_fig = True
        except Exception:
            has_fig = False

    if has_fig:
        md_lines.extend([
            "",
            "## 2. Biểu Đồ So Sánh",
            "",
            "![benchmark_comparison](figures/benchmark_comparison.png)",
        ])

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    return {
        "markdown": md_path,
        "figure": fig_path if has_fig else None,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark summary report from JSON results.")
    default_dir = Path("reports") if Path("reports").exists() else Path("reconia/reports")
    parser.add_argument("--report-dir", type=Path, default=default_dir, help="Path to reports directory")
    args = parser.parse_args()
    summary = generate_benchmark_summary(args.report_dir)
    print(f"[SUCCESS] Markdown summary: {summary['markdown']}")
    print(f"[SUCCESS] Figure comparison: {summary['figure']}")

