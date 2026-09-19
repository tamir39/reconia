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
        "tok_micro_f1",
        "tok_macro_f1",
        "ent_micro_f1",
        "ent_macro_f1",
        "latency_ms",
        "peak_vram_mb",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


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
    best_tok_macro = summary_data.get("best_tok_macro_f1", 0.0)
    best_tok_micro = summary_data.get("best_tok_micro_f1", 0.0)
    best_ent_f1 = summary_data.get("best_ent_micro_f1", 0.0)
    total_params = summary_data.get("total_params", 0)
    latency_ms = summary_data.get("latency_ms", 0.0)
    peak_vram = summary_data.get("peak_vram_mb", 0.0)
    device = summary_data.get("device", "CPU")
    history = summary_data.get("history", [])
    per_class = summary_data.get("final_per_class", {})

    lines: list[str] = [
        f"# Báo Cáo Thực Nghiệm: {model_name} trên Bộ Dữ Liệu {dataset_name}",
        "",
        "## 1. Tóm Tắt Kết Quả Tối Ưu Nhất",
        "",
        "| Tiêu chí đánh giá | Giá trị đạt được |",
        "|---|---|",
        f"| **Mô hình** | `{model_name}` |",
        f"| **Bộ dữ liệu** | `{dataset_name}` |",
        f"| **Thiết bị phần cứng** | `{device}` |",
        f"| **Tổng số tham số** | {total_params:,} |",
        f"| **Epoch tốt nhất** | Epoch {best_epoch} |",
        f"| **Token Macro-F1 (Tối ưu)** | **{best_tok_macro * 100:.2f}%** |",
        f"| **Token Micro-F1** | **{best_tok_micro * 100:.2f}%** |",
        f"| **Entity Micro-F1** | **{best_ent_f1 * 100:.2f}%** |",
        f"| **Độ trễ suy luận trung bình** | **{latency_ms:.2f} ms / tài liệu** |",
        f"| **Bộ nhớ GPU đỉnh điểm (VRAM)** | **{peak_vram:.1f} MB** |",
        "",
    ]

    # Embedded Figures
    if figure_paths:
        lines.append("## 2. Biểu Đồ Huấn Luyện & Đánh Giá")
        lines.append("")
        for fig_p in figure_paths:
            rel_path = fig_p.name
            lines.append(f"![{fig_p.stem}]({rel_path})")
            lines.append("")

    # Per-class Table
    if per_class:
        lines.append("## 3. Đánh Giá Chi Tiết Theo Từng Lớp Thực Thể (Per-Class Breakdown)")
        lines.append("")
        lines.append("| Tên nhãn BIO | Precision (%) | Recall (%) | F1-Score (%) | Số lượng mẫu (Support) |")
        lines.append("|---|---|---|---|---|")
        for cls_name, stats in sorted(per_class.items()):
            p = stats.get("precision", 0.0) * 100
            r = stats.get("recall", 0.0) * 100
            f1 = stats.get("f1", 0.0) * 100
            supp = stats.get("support", 0)
            lines.append(f"| `{cls_name}` | {p:.2f}% | {r:.2f}% | **{f1:.2f}%** | {supp:,} |")
        lines.append("")

    # History Table
    if history:
        lines.append("## 4. Lịch Sử Huấn Luyện Từng Epoch (Training History)")
        lines.append("")
        lines.append("| Epoch | Train Loss | Val Loss | Tok Micro-F1 | Tok Macro-F1 | Ent Micro-F1 | Latency (ms) | Peak VRAM |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for h in history:
            ep = h.get("epoch", 0)
            tl = h.get("train_loss", 0.0)
            vl = h.get("val_loss", 0.0)
            t_mic = h.get("tok_micro_f1", 0.0) * 100
            t_mac = h.get("tok_macro_f1", 0.0) * 100
            e_mic = h.get("ent_micro_f1", 0.0) * 100
            lat = h.get("latency_ms", 0.0)
            vram = h.get("peak_vram_mb", 0.0)
            lines.append(f"| {ep} | {tl:.4f} | {vl:.4f} | {t_mic:.2f}% | {t_mac:.2f}% | {e_mic:.2f}% | {lat:.1f} ms | {vram:.1f} MB |")
        lines.append("")

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def plot_training_dashboard(
    figure_path: Path,
    history: list[dict[str, Any]],
    per_class_metrics: dict[str, dict[str, float]],
    model_name: str,
    dataset_name: str,
) -> Path | None:
    """Generate a high-resolution 4-panel training and evaluation dashboard figure."""
    if plt is None:
        print("[Warning] matplotlib is not installed. Skipping chart generation.")
        return None

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return None

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    tok_micro = [h["tok_micro_f1"] * 100 for h in history]
    tok_macro = [h["tok_macro_f1"] * 100 for h in history]
    ent_micro = [h["ent_micro_f1"] * 100 for h in history]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle(
        f"Training & Evaluation Metrics: {model_name} on {dataset_name.upper()}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, "o-", color="#1f77b4", label="Train Loss", linewidth=2, markersize=5)
    ax1.plot(epochs, val_loss, "s--", color="#d62728", label="Val Loss", linewidth=2, markersize=5)
    ax1.set_title("Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Token F1 Curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, tok_macro, "^-", color="#2ca02c", label="Token Macro-F1 (%)", linewidth=2, markersize=5)
    ax2.plot(epochs, tok_micro, "v--", color="#ff7f0e", label="Token Micro-F1 (%)", linewidth=2, markersize=5)
    ax2.set_title("Token-Level Classification F1", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("F1 Score (%)", fontsize=10)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Entity F1 Curve
    ax3 = axes[1, 0]
    ax3.plot(epochs, ent_micro, "D-", color="#9467bd", label="Entity Micro-F1 (%)", linewidth=2, markersize=5)
    ax3.set_title("Entity-Level Extraction F1 (Exact Match)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Entity F1 (%)", fontsize=10)
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Per-Class F1 Bar Chart (Filtering out background 'O')
    ax4 = axes[1, 1]
    entity_classes = [
        (cls, stats["f1"] * 100)
        for cls, stats in per_class_metrics.items()
        if cls != "O" and stats.get("support", 0) > 0
    ]
    # Sort by F1 descending
    entity_classes.sort(key=lambda x: x[1], reverse=True)
    # Cap to top 15 if there are too many classes (like DocILE)
    if len(entity_classes) > 15:
        entity_classes = entity_classes[:15]

    if entity_classes:
        names = [item[0] for item in entity_classes]
        scores = [item[1] for item in entity_classes]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax4.barh(names, scores, color=colors, edgecolor="black", linewidth=0.5)
        ax4.set_title("Per-Class F1 Score Breakdown (Top Classes)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("F1 Score (%)", fontsize=10)
        ax4.set_xlim(0, 100)
        ax4.invert_yaxis()  # Highest at top
        for bar in bars:
            width = bar.get_width()
            ax4.text(
                width + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                fontsize=8,
            )
    else:
        ax4.text(0.5, 0.5, "No active entity classes", ha="center", va="center")
        ax4.set_title("Per-Class Breakdown", fontsize=12, fontweight="bold")

    ax4.grid(True, linestyle="--", alpha=0.6, axis="x")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
) -> dict[str, Path]:
    """Orchestrate saving JSON, CSV, Markdown, and Chart figures.

    Returns dict mapping format names ('json', 'csv', 'markdown', 'figure') to file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{model_name.lower()}_{dataset_name.lower()}"
    json_path = output_dir / f"{base_name}_results.json"
    csv_path = output_dir / f"{base_name}_history.csv"
    md_path = output_dir / f"{base_name}_report.md"
    fig_path = figures_dir / f"{base_name}_training_curves.png"

    # Find best epoch based on token macro F1
    best_entry = max(history, key=lambda x: x.get("tok_macro_f1", 0.0)) if history else {}
    best_epoch = best_entry.get("epoch", 1)

    summary_data = {
        "model": model_name,
        "dataset": dataset_name,
        "device": metadata.get("device", "CPU"),
        "total_params": metadata.get("total_params", 0),
        "best_epoch": best_epoch,
        "best_tok_macro_f1": best_entry.get("tok_macro_f1", 0.0),
        "best_tok_micro_f1": best_entry.get("tok_micro_f1", 0.0),
        "best_ent_micro_f1": best_entry.get("ent_micro_f1", 0.0),
        "latency_ms": best_entry.get("latency_ms", 0.0),
        "peak_vram_mb": best_entry.get("peak_vram_mb", 0.0),
        "final_per_class": final_val_metrics.get("token_metrics", {}).get("per_class", {}),
        "history": history,
        "metadata": metadata,
    }

    # 1. Save JSON
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_data, fh, indent=2, ensure_ascii=False)

    # 2. Save CSV
    save_training_history_csv(csv_path, history)

    # 3. Generate Chart Figure
    generated_fig = plot_training_dashboard(
        fig_path,
        history,
        summary_data["final_per_class"],
        model_name,
        dataset_name,
    )

    # 4. Save Markdown Report with embedded chart
    fig_list = [generated_fig] if generated_fig else []
    save_markdown_report(md_path, summary_data, figure_paths=fig_list)

    # 5. Automatically update global cross-model benchmark summary & comparative charts
    summary_res = generate_benchmark_summary(output_dir)

    result_dict: dict[str, Path] = {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "figure": fig_path,
    }
    if summary_res.get("markdown"):
        result_dict["benchmark_summary"] = summary_res["markdown"]
    if summary_res.get("figure"):
        result_dict["benchmark_figure"] = summary_res["figure"]

    return result_dict



def generate_benchmark_summary(reports_dir: Path) -> dict[str, Path | None]:
    """Scan all *_results.json in reports_dir, build comparative tables and charts.

    Exports:
      - reports/benchmark_summary.md (comprehensive cross-model comparison table)
      - reports/figures/benchmark_comparison.png (grouped bar charts of F1 and Latency/VRAM)
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

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
        "# Bảng So Sánh Tổng Hợp Các Mô Hình Học Sâu",
        "",
        "Báo cáo tự động tổng hợp từ các kết quả thực nghiệm trong thư mục `reports/`.",
        "",
        "## 1. Bảng So Sánh Toàn Diện (Comprehensive Benchmark Table)",
        "",
        "| Mô hình | Bộ dữ liệu | Epoch tối ưu | Token Macro-F1 | Token Micro-F1 | Entity Micro-F1 | Độ trễ (ms/doc) | GPU VRAM (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in records:
        model = r.get("model", "Unknown")
        dataset = r.get("dataset", "Unknown").upper()
        best_ep = r.get("best_epoch", 1)
        tok_macro = r.get("best_tok_macro_f1", 0.0) * 100
        tok_micro = r.get("best_tok_micro_f1", 0.0) * 100
        ent_micro = r.get("best_ent_micro_f1", 0.0) * 100
        lat = r.get("latency_ms", 0.0)
        vram = r.get("peak_vram_mb", 0.0)

        md_lines.append(
            f"| **{model}** | `{dataset}` | Epoch {best_ep} | **{tok_macro:.2f}%** | {tok_micro:.2f}% | **{ent_micro:.2f}%** | {lat:.2f} ms | {vram:.1f} MB |"
        )

    fig_path = figures_dir / "benchmark_comparison.png"
    has_fig = False

    # 2. Generate Comparative Bar Chart if matplotlib is available
    if plt is not None and len(records) >= 1:
        try:
            labels = [f"{r.get('model', '')}\n({r.get('dataset', '').upper()})" for r in records]
            tok_macro_scores = [r.get("best_tok_macro_f1", 0.0) * 100 for r in records]
            ent_micro_scores = [r.get("best_ent_micro_f1", 0.0) * 100 for r in records]
            latencies = [r.get("latency_ms", 0.0) for r in records]
            vrams = [r.get("peak_vram_mb", 0.0) for r in records]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
            fig.suptitle("Đánh Giá So Sánh Thực Nghiệm Các Mô Hình Học Sâu", fontsize=14, fontweight="bold")

            x = np.arange(len(labels))
            width = 0.35

            # Subplot 1: Accuracy (Macro-F1 & Entity Micro-F1)
            ax1 = axes[0]
            rects1 = ax1.bar(x - width / 2, tok_macro_scores, width, label="Token Macro-F1", color="#1f77b4")
            rects2 = ax1.bar(x + width / 2, ent_micro_scores, width, label="Entity Micro-F1", color="#ff7f0e")
            ax1.set_title("Độ Chính Xác Trích Xuất (F1-Score)", fontsize=11, fontweight="bold")
            ax1.set_ylabel("F1 Score (%)", fontsize=10)
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, fontsize=9)
            ax1.set_ylim(0, 100)
            ax1.legend(loc="upper left")
            ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
            for rect in list(rects1) + list(rects2):
                h = rect.get_height()
                if h > 0:
                    ax1.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=8)

            # Subplot 2: Inference Latency
            ax2 = axes[1]
            bars_lat = ax2.bar(x, latencies, width=0.45, color="#2ca02c", edgecolor="black", linewidth=0.5)
            ax2.set_title("Độ Trễ Suy Luận (Inference Latency)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Thời gian (ms / tài liệu)", fontsize=10)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=9)
            ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
            for bar in bars_lat:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, h + (max(latencies) * 0.02 + 0.1), f"{h:.1f} ms", ha="center", fontsize=8)

            # Subplot 3: GPU Memory Footprint
            ax3 = axes[2]
            bars_vram = ax3.bar(x, vrams, width=0.45, color="#d62728", edgecolor="black", linewidth=0.5)
            ax3.set_title("Tiêu Thụ Bộ Nhớ GPU (Peak VRAM)", fontsize=11, fontweight="bold")
            ax3.set_ylabel("Dung lượng (MB)", fontsize=10)
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels, fontsize=9)
            ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
            for bar in bars_vram:
                h = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2, h + (max(vrams) * 0.02 + 1.0), f"{h:.1f} MB", ha="center", fontsize=8)

            plt.tight_layout()
            plt.savefig(fig_path, format="png", bbox_inches="tight")
            plt.close(fig)
            has_fig = True
        except Exception:
            has_fig = False

    if has_fig:
        md_lines.extend([
            "",
            "## 2. Biểu Đồ Trực Quan So Sánh (Comparative Charts)",
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
    parser.add_argument("--reports-dir", type=str, default="reconia/reports", help="Path to reports directory")
    args = parser.parse_args()
    summary = generate_benchmark_summary(Path(args.reports_dir))
    print(f"[SUCCESS] Markdown summary: {summary['markdown']}")
    print(f"[SUCCESS] Figure comparison: {summary['figure']}")

