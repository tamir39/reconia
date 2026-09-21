"""Run benchmark suite across multiple models and datasets.

Usage:
    python -m app.training.run_benchmark
    python -m app.training.run_benchmark --models roberta --datasets sroie
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.data.paths import WORKSPACE_ROOT
from app.evaluation.report import generate_benchmark_summary

RECONIA_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable


@dataclass
class ModelBenchmarkSpec:
    name: str
    entry_module: str
    default_lr: float
    extra_args: list[str] = field(default_factory=list)


MODEL_SPECS: dict[str, ModelBenchmarkSpec] = {
    "gnn": ModelBenchmarkSpec(
        name="gnn",
        entry_module="app.training.train_gnn",
        default_lr=1e-3,
        extra_args=[],
    ),
    "roberta": ModelBenchmarkSpec(
        name="roberta",
        entry_module="app.training.train_roberta",
        default_lr=5e-5,
        extra_args=["--head-lr", "2e-4"],
    ),
}

SUPPORTED_MODELS = list(MODEL_SPECS.keys())


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def run_benchmark_suite(
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    epochs: int = 20,
    patience: int = 5,
    lr: float | None = None,
    batch_accum: int | None = None,
    monitor_metric: str = "f1",
    limit_train: int | None = None,
    limit_val: int | None = None,
    device: str = "auto",
    reports_dir: Path | None = None,
    save_dir: Path | None = None,
) -> int:
    reports_dir = reports_dir or RECONIA_ROOT / "reports"
    save_dir = save_dir or RECONIA_ROOT / "checkpoints"
    models = [m for m in (models or list(SUPPORTED_MODELS)) if m in SUPPORTED_MODELS] or list(SUPPORTED_MODELS)
    datasets = datasets or ["sroie", "cord", "docile"]

    reports_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = reports_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"benchmark_run_{timestamp_str}.log"

    def log_and_print(msg: str) -> None:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    total_experiments = len(models) * len(datasets)

    log_and_print("=" * 80)
    log_and_print(f"=== RECONIA BENCHMARK SUITE (TRAINING & EVALUATION) ===")
    log_and_print(f"Start Time:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"Models:       {', '.join(m.upper() for m in models)}")
    log_and_print(f"Datasets:     {', '.join(d.upper() for d in datasets)}")
    log_and_print(f"Experiments:  {total_experiments} total runs")
    log_and_print(f"Epochs:       {epochs} (Patience: {patience})")
    if limit_train or limit_val:
        log_and_print(f"Limit:        Train={limit_train or 'Full'}, Val={limit_val or 'Full'}")
    log_and_print(f"Python:       {PYTHON_EXE}")
    log_and_print(f"Reports Dir:  {reports_dir}")
    log_and_print(f"Log File:     {log_file}")
    log_and_print("=" * 80)

    overall_start = time.time()
    results_summary: list[dict[str, str]] = []
    exp_counter = 0

    for model in models:
        spec = MODEL_SPECS.get(
            model,
            ModelBenchmarkSpec(
                name=model,
                entry_module=f"app.training.train_{model}",
                default_lr=1e-4,
                extra_args=[],
            ),
        )
        chosen_lr = lr if lr is not None else spec.default_lr

        for dataset in datasets:
            exp_counter += 1
            log_and_print(
                f"\n[{exp_counter}/{total_experiments}] Training {model.upper()} on {dataset.upper()} "
                f"at {datetime.datetime.now().strftime('%H:%M:%S')}..."
            )
            task_start = time.time()

            cmd = [
                PYTHON_EXE,
                "-u",
                "-m",
                spec.entry_module,
                "--dataset",
                dataset,
                "--epochs",
                str(epochs),
                "--patience",
                str(patience),
                "--lr",
                str(chosen_lr),
                "--device",
                device,
                "--save-dir",
                str(save_dir),
                "--report-dir",
                str(reports_dir),
                "--monitor-metric",
                monitor_metric,
            ]
            cmd.extend(spec.extra_args)

            if batch_accum is not None:
                cmd.extend(["--batch-accum", str(batch_accum)])

            if limit_train is not None:
                cmd.extend(["--limit-train", str(limit_train)])
            if limit_val is not None:
                cmd.extend(["--limit-val", str(limit_val)])

            log_and_print(f"Command: {' '.join(cmd)}")

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(RECONIA_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                assert process.stdout is not None
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        line_str = line.rstrip("\n\r")
                        log_and_print(f"  [{model.upper()}-{dataset.upper()}] {line_str}")

                process.wait()
                ret_code = process.returncode
            except Exception as e:
                log_and_print(f"[ERROR] Exception occurred while executing {model} on {dataset}: {e}")
                ret_code = -1

            task_elapsed = time.time() - task_start
            status_str = "SUCCESS" if ret_code == 0 else f"FAILED (code {ret_code})"
            log_and_print(
                f"[{exp_counter}/{total_experiments}] Completed {model.upper()} on {dataset.upper()} "
                f"in {format_duration(task_elapsed)} -> {status_str}"
            )

            results_summary.append({
                "model": model.upper(),
                "dataset": dataset.upper(),
                "status": status_str,
                "duration": format_duration(task_elapsed),
            })

            # Update global comparative benchmark summary after every single experiment (full training only)
            if limit_train is None and limit_val is None:
                log_and_print(f"Updating comparative benchmark charts in {reports_dir}...")
                try:
                    generate_benchmark_summary(reports_dir)
                except Exception as e:
                    log_and_print(f"Warning: Failed to update comparative summary: {e}")

    total_elapsed = time.time() - overall_start
    log_and_print("\n" + "=" * 80)
    log_and_print(f"[DONE] BENCHMARK PIPELINE COMPLETED at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"Total Elapsed Time: {format_duration(total_elapsed)}")
    log_and_print("-" * 80)
    log_and_print(f"{'Model':<12} | {'Dataset':<12} | {'Status':<18} | {'Duration':<15}")
    log_and_print("-" * 80)
    for res in results_summary:
        log_and_print(f"{res['model']:<12} | {res['dataset']:<12} | {res['status']:<18} | {res['duration']:<15}")
    log_and_print("=" * 80)
    log_and_print(f"Reports & charts generated at: {reports_dir}")
    log_and_print(f"Benchmark summary:             {reports_dir / 'benchmark_summary.md'}")
    log_and_print(f"Comparative figure:            {reports_dir / 'figures' / 'benchmark_comparison.png'}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to benchmark (default: all; or specify model names)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "sroie", "cord", "docile"],
        help="Datasets to train sequentially: 'all' (default: sroie cord docile), 'sroie', 'cord', 'docile'",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per dataset")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (0 to disable)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: model-specific)")
    parser.add_argument("--batch-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument(
        "--monitor-metric",
        default="f1",
        choices=["f1", "val_loss"],
        help="Metric to monitor for best checkpoint and early stopping: 'f1' (default) or 'val_loss'",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Limit train docs per dataset (default: None for full dataset; use for fast debugging)",
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=None,
        help="Limit val docs per dataset (default: None for full dataset; use for fast debugging)",
    )
    parser.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory for benchmark reports")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory for model checkpoints")
    args = parser.parse_args(argv)

    # Process models
    if "all" in args.models:
        selected_models = list(SUPPORTED_MODELS)
    else:
        selected_models = [m.lower() for m in args.models if m.lower() in SUPPORTED_MODELS]
        if not selected_models:
            selected_models = ["gnn"]

    # Process datasets
    if "all" in args.datasets:
        selected_datasets = ["sroie", "cord", "docile"]
    else:
        selected_datasets = [d.lower() for d in args.datasets if d.lower() in ("sroie", "cord", "docile")]

    return run_benchmark_suite(
        models=selected_models,
        datasets=selected_datasets,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_accum=args.batch_accum,
        monitor_metric=args.monitor_metric,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        device=args.device,
        reports_dir=args.report_dir,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
