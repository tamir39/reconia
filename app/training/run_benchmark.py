"""Automated benchmark pipeline for invoice information extraction models.

Executes sequential training across selected models (GNN, RoBERTa) and
datasets (SROIE, CORD-v2, DocILE), logs progress with timestamps, profiles hardware,
and automatically updates comparative benchmark reports and visualization dashboards.

Usage:
    python -m app.training.run_benchmark --models roberta --datasets sroie --epochs 5
    python -m app.training.run_benchmark --models all --all --epochs 10 --patience 5
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

from app.data.paths import WORKSPACE_ROOT
from app.evaluation.report import generate_benchmark_summary

RECONIA_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable

SUPPORTED_MODELS = ["gnn", "roberta"]
MODEL_DEFAULT_LR = {
    "gnn": 1e-3,
    "roberta": 3e-5,
}


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def run_benchmark_suite(
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    epochs: int = 10,
    patience: int = 5,
    lr: float | None = None,
    batch_accum: int | None = None,
    docile_limit_train: int | None = None,
    docile_limit_val: int | None = None,
    device: str = "auto",
    reports_dir: Path | None = None,
    save_dir: Path | None = None,
) -> int:
    reports_dir = reports_dir or RECONIA_ROOT / "reports"
    save_dir = save_dir or RECONIA_ROOT / "checkpoints"
    models = models or ["gnn"]
    datasets = datasets or ["sroie", "cord", "docile"]

    reports_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = reports_dir / f"benchmark_run_{timestamp_str}.log"

    def log_and_print(msg: str) -> None:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    total_experiments = len(models) * len(datasets)

    log_and_print("=" * 80)
    log_and_print("=== RECONIA AUTOMATED BENCHMARK PIPELINE ===")
    log_and_print(f"Start Time:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"Models:       {', '.join(m.upper() for m in models)}")
    log_and_print(f"Datasets:     {', '.join(d.upper() for d in datasets)}")
    log_and_print(f"Experiments:  {total_experiments} total runs")
    log_and_print(f"Epochs:       {epochs} (Patience: {patience})")
    log_and_print(f"Python:       {PYTHON_EXE}")
    log_and_print(f"Reports Dir:  {reports_dir}")
    log_and_print(f"Log File:     {log_file}")
    log_and_print("=" * 80)

    overall_start = time.time()
    results_summary: list[dict[str, str]] = []
    exp_counter = 0

    for model in models:
        # Determine model-appropriate learning rate
        chosen_lr = lr if lr is not None else MODEL_DEFAULT_LR.get(model, 1e-4)

        for dataset in datasets:
            exp_counter += 1
            log_and_print(
                f"\n[{exp_counter}/{total_experiments}] Starting {model.upper()} on {dataset.upper()} "
                f"(LR: {chosen_lr}) at {datetime.datetime.now().strftime('%H:%M:%S')}..."
            )
            task_start = time.time()

            cmd = [
                PYTHON_EXE,
                "-m",
                f"app.training.train_{model}",
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
            ]

            if batch_accum is not None:
                cmd.extend(["--batch-accum", str(batch_accum)])

            if dataset == "docile":
                if docile_limit_train:
                    cmd.extend(["--limit-train", str(docile_limit_train)])
                if docile_limit_val:
                    cmd.extend(["--limit-val", str(docile_limit_val)])

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
                for line in process.stdout:
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

            # Update global comparative benchmark summary after every single experiment
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
        default=["gnn"],
        help="Models to benchmark: 'gnn', 'roberta', or 'all' (default: gnn)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sroie", "cord"],
        choices=["sroie", "cord", "docile"],
        help="Datasets to train on sequentially (default: sroie cord)",
    )
    parser.add_argument("--all", action="store_true", help="Run on all 3 datasets (sroie, cord, docile)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per dataset")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (0 to disable)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: model-specific)")
    parser.add_argument("--batch-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument(
        "--docile-limit-train",
        type=int,
        default=500,
        help="Limit train docs for DocILE (default: 500 for fast benchmark; pass 0 for full 5180 docs)",
    )
    parser.add_argument(
        "--docile-limit-val",
        type=int,
        default=100,
        help="Limit val docs for DocILE (default: 100; pass 0 for full 500 docs)",
    )
    parser.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    # Process models
    if "all" in args.models:
        selected_models = list(SUPPORTED_MODELS)
    else:
        selected_models = [m.lower() for m in args.models if m.lower() in SUPPORTED_MODELS]
        if not selected_models:
            selected_models = ["gnn"]

    datasets = ["sroie", "cord", "docile"] if args.all else args.datasets
    docile_train = None if args.docile_limit_train == 0 else args.docile_limit_train
    docile_val = None if args.docile_limit_val == 0 else args.docile_limit_val

    return run_benchmark_suite(
        models=selected_models,
        datasets=datasets,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_accum=args.batch_accum,
        docile_limit_train=docile_train,
        docile_limit_val=docile_val,
        device=args.device,
        reports_dir=args.reports_dir,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
