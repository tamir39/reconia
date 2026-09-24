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
    provenance = summary_data.get("evaluation_provenance", {})
    if not provenance.get("protocol"):
        lines.extend(["> Kết quả lịch sử chưa có đủ thông tin để tái lập; cần chạy lại quy trình đánh giá mới.", ""])
    else:
        lines.extend([f"Giao thức: `{provenance['protocol']}`. Đây là chỉ số nội bộ, không phải điểm DocILE chính thức.", ""])
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
    if test_metrics is not None and test_metrics.get("complete") is False:
        raise ValueError("A partial prediction run cannot be published as a full test benchmark")
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
    primary_source = test_metrics if test_metrics is not None else final_val_metrics
    per_type = (
        primary_source.get("per_type")
        or primary_source.get("entity_metrics", {}).get("per_type")
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
        "evaluation_provenance": {
            key: primary_source[key] for key in (
                "protocol", "complete", "num_documents", "expected_num_documents",
                "split_sha256", "checkpoint_sha256", "prediction_sha256",
            ) if key in primary_source
        },
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
    if any(not r.get("evaluation_provenance", {}).get("protocol") for r in records):
        md_lines[2:2] = [
            "> Có kết quả lịch sử chưa được tái lập bằng quy trình mới. Không dùng bảng này làm kết luận cuối cùng.",
            "",
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
            datasets = sorted(list(dict.fromkeys(r.get("dataset", "Unknown").upper() for r in records)))
            models = sorted(list(dict.fromkeys(r.get("model", "Unknown") for r in records)))
            rec_map = {(r.get("dataset", "Unknown").upper(), r.get("model", "Unknown")): r for r in records}

            num_datasets = len(datasets)
            num_models = len(models)

            model_palette = {
                "GNN": "#2ca02c",         # Green
                "RoBERTa": "#1f77b4",     # Blue
                "LayoutLMv3": "#d62728",  # Red
                "LiLT": "#8c564b",        # Brown
            }
            default_colors = [
                "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"
            ]

            fig_width = max(10.5, num_datasets * num_models * 0.85 + 2.0)
            fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5.5), dpi=300)
            fig.suptitle("Model Performance Comparison (TEST Set)", fontsize=14, fontweight="bold", y=0.98)

            total_group_width = 0.80
            bar_width = total_group_width / max(num_models, 1)

            ax1, ax2 = axes[0], axes[1]

            # Subplot 1: F1-Score
            ax1.set_title("F1-Score (%)", fontsize=11, fontweight="bold", pad=10)
            ax1.set_ylabel("F1-Score (%)", fontsize=10, fontweight="semibold")
            ax1.set_ylim(0, 115)
            ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

            # Subplot 2: Latency
            ax2.set_title("Latency (ms)", fontsize=11, fontweight="bold", pad=10)
            ax2.set_ylabel("Average Latency (ms)", fontsize=10, fontweight="semibold")
            max_lat = max((float(r.get("latency", 0.0)) for r in records), default=1.0)
            ax2.set_ylim(0, max_lat * 1.28 + 2.0)
            ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

            x = np.arange(num_datasets)
            val_font = 7.0 if num_models >= 4 else 8.5

            for j, m in enumerate(models):
                offset = (j - (num_models - 1) / 2) * bar_width
                color = model_palette.get(m, default_colors[j % len(default_colors)])

                for i, d in enumerate(datasets):
                    rec = rec_map.get((d, m))
                    if rec is None:
                        continue

                    f1_val = float(rec.get("f1", 0.0)) * 100
                    lat_val = float(rec.get("latency", 0.0))
                    pos = i + offset

                    ax1.bar(pos, f1_val, width=bar_width * 0.88, label=m if i == 0 else "", color=color, edgecolor="black", linewidth=0.6, alpha=0.9)
                    ax2.bar(pos, lat_val, width=bar_width * 0.88, label=m if i == 0 else "", color=color, edgecolor="black", linewidth=0.6, alpha=0.9)

                    if f1_val > 0:
                        ax1.text(pos, f1_val + 1.5, f"{f1_val:.1f}%", ha="center", va="bottom", fontsize=val_font, fontweight="semibold", rotation=0)
                    if lat_val > 0:
                        lat_text = f"{lat_val:.1f}" if num_models >= 4 else f"{lat_val:.1f} ms"
                        ax2.text(pos, lat_val + (max_lat * 0.015 + 0.2), lat_text, ha="center", va="bottom", fontsize=val_font, fontweight="semibold", rotation=0)

            for ax in [ax1, ax2]:
                ax.set_xticks(x)
                ax.set_xticklabels(datasets, fontsize=10.5, fontweight="bold")
                ax.tick_params(axis="x", pad=6)

            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=model_palette.get(m, default_colors[j % len(default_colors)]), edgecolor="black", linewidth=0.5, label=m)
                for j, m in enumerate(models)
            ]
            ncol_val = min(len(models), 6)
            fig.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.935),
                ncol=ncol_val,
                frameon=True,
                facecolor="#f9f9f9",
                edgecolor="#cccccc",
                fontsize=9,
                title="Model",
                title_fontsize=9.5,
            )

            plt.tight_layout(rect=[0, 0.04, 1, 0.86])
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

