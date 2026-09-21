"""Training pipeline for the Spatial Relational GNN model (Model 1).

Supports all 3 benchmark datasets (SROIE, CORD-v2, DocILE).
Profiles inference latency, throughput, token metrics, and entity metrics.

Usage:
    python -m app.training.train_gnn --dataset sroie --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    torch = None

from app.data import CordAdapter, DocileAdapter, SroieAdapter
from app.data.schema import DatasetAdapter, Document
from app.evaluation import (
    InferenceProfiler,
    compute_entity_metrics,
    compute_token_metrics,
    export_full_benchmark_report,
    extract_entities_from_bio,
    save_training_history_csv,
)
from app.graph.builder import DocumentGraph
from app.graph.cache import BuildConfig, GraphCache, build_vocabulary, load_vocabulary, save_vocabulary
from app.graph.edges import compute_spatial_edge_features
from app.models_dl.gnn import InvoiceGNN, InvoiceGNNConfig

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache"
DEFAULT_CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"
DEFAULT_REPORTS = Path(__file__).resolve().parents[2] / "reports"

ADAPTERS: dict[str, type] = {
    "sroie": SroieAdapter,
    "cord": CordAdapter,
    "docile": DocileAdapter,
}

from app.training.utils import DualLogger, str2bool


def prepare_graph_tensors(
    graph: DocumentGraph,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a DocumentGraph into PyTorch tensors on the target device."""
    u_edge_index, u_edge_type = graph.to_undirected()
    spatial_attr = compute_spatial_edge_features(graph.features, u_edge_index)

    geo_tensor = torch.tensor(graph.features, dtype=torch.float32, device=device)
    edge_index_tensor = torch.tensor(u_edge_index, dtype=torch.long, device=device)
    edge_type_tensor = torch.tensor(u_edge_type, dtype=torch.long, device=device)
    edge_attr_tensor = torch.tensor(spatial_attr, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(graph.labels, dtype=torch.long, device=device)

    return (
        geo_tensor,
        graph.texts,
        edge_index_tensor,
        edge_type_tensor,
        edge_attr_tensor,
        labels_tensor,
    )


def train_one_epoch(
    model: nn.Module,
    cache: GraphCache,
    train_docs: list[Document],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_accum: int = 8,
    clip_grad_norm: float = 1.0,
) -> float:
    """Train model for one epoch using gradient accumulation across documents."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    accum_steps = 0

    # Shuffle documents per epoch
    indices = np.random.permutation(len(train_docs))

    for step, idx in enumerate(indices, start=1):
        doc = train_docs[idx]
        graph = cache.get(doc)
        if graph.num_nodes == 0:
            continue

        geo, texts, edge_idx, edge_t, edge_a, targets = prepare_graph_tensors(graph, device)

        logits = model(geo, texts, edge_idx, edge_t, edge_a)
        loss = criterion(logits, targets) / batch_accum

        loss.backward()
        total_loss += loss.item() * batch_accum
        accum_steps += 1

        if accum_steps % batch_accum == 0 or step == len(indices):
            # Gradient clipping for training stability
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(len(train_docs), 1)



@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    cache: GraphCache,
    val_docs: list[Document],
    criterion: nn.Module,
    device: torch.device,
    vocabulary: dict[str, int],
) -> dict[str, Any]:
    """Evaluate model on validation split, computing token and entity metrics."""
    model.eval()
    total_loss = 0.0

    all_preds: list[int] = []
    all_targets: list[int] = []

    pred_spans_batch = []
    true_spans_batch = []

    id_to_tag = {v: k for k, v in vocabulary.items()}

    profiler = InferenceProfiler(device=str(device))
    profiler.start()

    for doc in val_docs:
        graph = cache.get(doc)
        if graph.num_nodes == 0:
            continue

        geo, texts, edge_idx, edge_t, edge_a, targets = prepare_graph_tensors(graph, device)

        logits = model(geo, texts, edge_idx, edge_t, edge_a)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        targets_np = targets.cpu().numpy().tolist()

        all_preds.extend(preds)
        all_targets.extend(targets_np)

        pred_tags = [id_to_tag.get(p, "O") for p in preds]
        true_tags = [id_to_tag.get(t, "O") for t in targets_np]

        pred_spans = extract_entities_from_bio(pred_tags, graph.texts, graph.pages)
        true_spans = extract_entities_from_bio(true_tags, graph.texts, graph.pages)

        pred_spans_batch.append(pred_spans)
        true_spans_batch.append(true_spans)

    perf_stats = profiler.stop(num_documents=len(val_docs))

    # Token-level metrics
    token_metrics = compute_token_metrics(
        all_targets, all_preds, id_to_tag, ignore_labels={"O"}
    )

    # Entity-level metrics
    entity_metrics = compute_entity_metrics(
        pred_spans_batch, true_spans_batch, match_mode="exact"
    )

    avg_loss = total_loss / max(len(val_docs), 1)

    return {
        "val_loss": avg_loss,
        "precision": float(entity_metrics["precision"]),
        "recall": float(entity_metrics["recall"]),
        "f1": float(entity_metrics["f1"]),
        "entity_metrics": entity_metrics,
        "token_metrics": token_metrics,
        "perf": perf_stats,
    }


def compute_class_weights(
    train_docs: list[Document],
    vocabulary: dict[str, int],
    cache: GraphCache,
    power: float = 0.5,
) -> torch.Tensor:
    """Compute class frequency weights to counteract the severe dominance of 'O' tags."""
    counts = np.zeros(len(vocabulary), dtype=np.float32)
    for doc in train_docs[:100]:
        graph = cache.get(doc)
        for label in graph.labels:
            if 0 <= label < len(counts):
                counts[label] += 1
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / (counts**power)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.2, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


def main(argv: list[str] | None = None) -> int:
    if torch is None:
        print("ERROR: PyTorch is required to run train_gnn.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "sroie", "cord", "docile"],
        help="Dataset name: 'all' (default: sroie cord docile), 'sroie', 'cord', 'docile'",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay factor (default: 1e-4)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for GNN layers")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of EdgeConv GNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--batch-accum", type=int, default=16, help="Gradient accumulation steps (effective batch size = 16)")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping (<= 0 to disable)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs (0 to disable)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument(
        "--monitor-metric",
        default="f1",
        choices=["f1", "val_loss"],
        help="Metric to monitor for best checkpoint and early stopping: 'f1' (default) or 'val_loss'",
    )
    parser.add_argument(
        "--weighted-loss",
        type=str2bool,
        default=True,
        help="Enable class-frequency weighted loss in CrossEntropy",
    )
    parser.add_argument("--device", default="auto", help="Execution device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Path to precomputed feature/graph cache")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_CHECKPOINTS, help="Directory to save model checkpoints")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS, help="Directory to save logs, metrics, and figures")
    parser.add_argument("--limit-train", type=int, default=None, help="Limit number of training documents (debugging)")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit number of validation documents (debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    target_datasets = ["sroie", "cord", "docile"] if args.dataset == "all" else [args.dataset]
    for current_dataset in target_datasets:
        train_single_gnn_dataset(args, current_dataset, device)
    return 0


def train_single_gnn_dataset(args: argparse.Namespace, dataset_name: str, device: torch.device) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.report_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_gnn_{dataset_name}_{timestamp_str}.log"
    dual_logger = DualLogger(log_file)
    old_stdout = sys.stdout
    sys.stdout = dual_logger

    # 1. Load Dataset
    adapter_cls = ADAPTERS[dataset_name]
    adapter: DatasetAdapter = adapter_cls()

    train_docs = list(adapter.iter_documents("train", limit=args.limit_train))
    val_docs = list(adapter.iter_documents("val", limit=args.limit_val))

    # 2. Vocabulary setup
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.cache_dir / f"{dataset_name}-vocabulary.json"
    if vocab_path.exists():
        vocabulary, fieldtypes = load_vocabulary(vocab_path)
    else:
        vocabulary, fieldtypes = build_vocabulary(adapter.iter_documents("train"))
        save_vocabulary(vocab_path, vocabulary, fieldtypes)

    # 3. Graph Cache
    config = BuildConfig()
    cache = GraphCache(args.cache_dir, vocabulary, config)

    # 4. Model instantiation
    gnn_config = InvoiceGNNConfig(
        geo_dim=17,
        text_dim=128,
        edge_spatial_dim=6,
        num_edge_types=4,
        edge_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(vocabulary),
        dropout=args.dropout,
    )
    model = InvoiceGNN(gnn_config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loss & Optimizer
    if args.weighted_loss:
        class_weights = compute_class_weights(train_docs, vocabulary, cache).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    is_debug = bool(args.limit_train is not None or args.limit_val is not None)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = args.save_dir / f"gnn_{dataset_name}_best.pt"
    latest_checkpoint_path = args.save_dir / f"gnn_{dataset_name}_latest.pt"

    metrics_dir = args.report_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = metrics_dir / f"gnn_{dataset_name.lower()}_history.csv"

    start_epoch = 1
    best_epoch = 1
    best_score = float("inf") if args.monitor_metric == "val_loss" else -1.0
    best_val_metrics: dict[str, Any] = {}
    val_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    if args.resume is not None:
        if not args.resume.exists():
            print(f"ERROR: Checkpoint file not found: {args.resume}", file=sys.stderr)
            return 1
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        history = checkpoint.get("history", [])
        if "best_score" in checkpoint:
            best_score = checkpoint["best_score"]
        elif "metrics" in checkpoint:
            best_val_metrics = checkpoint["metrics"]
            if args.monitor_metric == "f1":
                best_score = best_val_metrics.get("f1", best_val_metrics.get("entity_metrics", {}).get("f1", -1.0))
            elif args.monitor_metric == "val_loss":
                best_score = best_val_metrics.get("val_loss", float("inf"))
        print(f"Successfully resumed at epoch {start_epoch}. Prior best {args.monitor_metric}: {best_score:.4f}")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}, but --epochs is set to {args.epochs}.\n"
            f"To train further, specify a larger --epochs (e.g. --epochs {start_epoch + 5})."
        )
        return 0

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
        last_epoch=(start_epoch - 2) if start_epoch > 1 else -1,
    )

    patience_counter = 0

    gpu_desc = f"{torch.cuda.get_device_name(0)}" if device.type == "cuda" else "CPU"
    loss_mode = "CrossEntropy (Weighted: True - Inverse class frequency)" if args.weighted_loss else "CrossEntropy (Weighted: False - Standard)"
    patience_desc = f"{args.patience} epochs (monitoring '{args.monitor_metric}')" if args.patience > 0 else "Disabled"
    fields_desc = f"{len(fieldtypes)} entity types ({', '.join(sorted(fieldtypes))})" if fieldtypes else "N/A"

    print(f"\n{'=' * 80}")
    print(f"{'RECONIA EXPERIMENT SETUP: SPATIAL RELATIONAL GNN':^80}")
    print(f"{'=' * 80}")
    print(f"[Hardware & Runtime]")
    print(f"  Device              : {str(device).upper()} ({gpu_desc})")
    print(f"  Random Seed         : {args.seed}")
    print(f"\n[Dataset & Labels]")
    print(f"  Target Dataset      : {dataset_name.upper()} (Train: {len(train_docs)} docs | Val: {len(val_docs)} docs)")
    print(f"  Vocabulary          : {len(vocabulary)} BIO tags across {fields_desc}")
    print(f"\n[Model Architecture]")
    print(f"  GNN Backbone        : EdgeConv GNN ({args.num_layers} layers, {args.hidden_dim} hidden, dropout: {args.dropout})")
    print(f"  Node & Edge Dims    : Geo: {gnn_config.geo_dim}-d, Text: {gnn_config.text_dim}-d, Spatial Edge: {gnn_config.edge_spatial_dim}-d, Edge Types: {gnn_config.num_edge_types}")
    print(f"  Parameters          : {total_params:,} total trainable parameters")
    print(f"\n[Training Hyperparameters]")
    print(f"  Batch Strategy      : Graph Batch: 1 x Accum: {args.batch_accum}  ==>  Effective Batch Size: {args.batch_accum}")
    print(f"  Learning Rate       : {args.lr:.1e} (CosineAnnealingLR, min: {args.lr * 0.05:.1e})")
    print(f"  Optimization        : AdamW (Weight Decay: {args.weight_decay} | Clip Norm: {args.clip_grad_norm})")
    print(f"  Schedule & EarlyStop: {args.epochs} max epochs (Patience: {patience_desc})")
    print(f"  Loss Criterion      : {loss_mode}")
    print(f"\n[Artifacts & Outputs]")
    print(f"  Best Checkpoint     : {best_checkpoint_path}")
    print(f"  Reports & Charts    : {args.report_dir}")
    print(f"{'=' * 80}")

    print("\n" + "=" * 80)
    header = f"{'Epoch':<6} | {'TrainLoss':<10} | {'ValLoss':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Latency(ms)':<11}"
    print(header)
    print("=" * 80)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model,
                cache,
                train_docs,
                optimizer,
                criterion,
                device,
                batch_accum=args.batch_accum,
                clip_grad_norm=args.clip_grad_norm,
            )
            scheduler.step()

            val_metrics = evaluate_model(model, cache, val_docs, criterion, device, vocabulary)
            val_loss = val_metrics["val_loss"]
            precision = val_metrics["precision"]
            recall = val_metrics["recall"]
            f1 = val_metrics["f1"]
            lat = val_metrics["perf"]["latency"]

            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "latency": float(lat),
            })

            # Real-time persistence of training history (full training only)
            if not is_debug:
                save_training_history_csv(history_csv_path, history)

            print(
                f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {precision:<10.4f} | "
                f"{recall:<10.4f} | {f1:<10.4f} | {lat:<11.1f}"
            )

            current_score = f1 if args.monitor_metric == "f1" else val_loss

            improved = (
                (current_score < best_score)
                if args.monitor_metric == "val_loss"
                else (current_score > best_score)
            )

            checkpoint_state = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": gnn_config,
                "vocabulary": vocabulary,
                "fieldtypes": fieldtypes,
                "metrics": val_metrics,
                "history": history,
                "best_score": current_score if improved else best_score,
                "monitor_metric": args.monitor_metric,
            }

            if improved:
                best_score = current_score
                best_epoch = epoch
                checkpoint_state["best_epoch"] = best_epoch
                best_val_metrics = val_metrics
                patience_counter = 0
                checkpoint_state["best_score"] = best_score
                torch.save(checkpoint_state, best_checkpoint_path)
            else:
                patience_counter += 1

            # Always save latest checkpoint for resume capability
            torch.save(checkpoint_state, latest_checkpoint_path)

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n[Early Stopping] No improvement in '{args.monitor_metric}' for {args.patience} consecutive epochs. Stopping at epoch {epoch}.")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Finalizing artifacts from completed epochs...")

    print("=" * 87)
    print(f"\nTraining completed. Best {args.monitor_metric}: {best_score:.4f} (at Epoch {best_epoch})")
    print(f"Best model checkpoint saved to: {best_checkpoint_path}")
    print(f"Latest checkpoint saved to:       {latest_checkpoint_path}")

    # Automatic Test Set Evaluation & Prediction Export on best checkpoint (Full Training only)
    test_metrics: dict[str, Any] | None = None
    if is_debug:
        print(f"\n[Debug Mode Detected] Skipping TEST set evaluation, prediction export, and benchmark reporting.")
    elif "test" in adapter.splits() and getattr(adapter, "has_ground_truth", lambda s: True)("test"):
        print(f"\n=== Evaluating Best Checkpoint & Exporting Predictions on TEST Set ({dataset_name.upper()}) ===")
        pred_dir = args.report_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"predictions_gnn_{dataset_name.lower()}_test.json"
        from app.evaluation.predict import run_prediction_pipeline

        test_metrics = run_prediction_pipeline(
            model_path=best_checkpoint_path,
            dataset_name=dataset_name,
            split="test",
            output_json=pred_file,
            device_str=str(device),
            cache_dir=args.cache_dir,
            model_type="gnn",
        )
        p_t = test_metrics["precision"] * 100
        r_t = test_metrics["recall"] * 100
        f1_t = test_metrics["f1"] * 100
        lat_t = test_metrics.get("latency", test_metrics.get("perf", {}).get("latency", 0.0))
        print(f"TEST Results (Best Checkpoint - Epoch {best_epoch}):")
        print(f"  Precision: {p_t:.2f}% | Recall: {r_t:.2f}% | F1: {f1_t:.2f}% | Latency: {lat_t:.1f} ms")
        print(f"  Predictions saved to: {pred_file}")

    # Export comprehensive benchmark reports (Full Training only)
    if not is_debug and (best_val_metrics or val_metrics) and history:
        metadata = {
            "device": str(device),
            "gpu_name": "CUDA GPU" if device.type == "cuda" else "CPU",
            "total_params": total_params,
            "trainable_params": total_params,
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "batch_accum": args.batch_accum,
            "effective_batch_size": args.batch_accum,
            "lr": args.lr,
            "patience": args.patience,
            "monitor_metric": args.monitor_metric,
            "clip_grad_norm": args.clip_grad_norm,
            "resumed_from": str(args.resume) if args.resume else None,
            "config": {
                "geo_dim": gnn_config.geo_dim,
                "text_dim": gnn_config.text_dim,
                "hidden_dim": gnn_config.hidden_dim,
                "num_layers": gnn_config.num_layers,
                "num_classes": gnn_config.num_classes,
                "dropout": gnn_config.dropout,
            },
            "seed": args.seed,
        }

        report_files = export_full_benchmark_report(
            output_dir=args.report_dir,
            model_name="GNN",
            dataset_name=dataset_name,
            history=history,
            final_val_metrics=best_val_metrics or val_metrics,
            test_metrics=test_metrics,
            metadata=metadata,
        )

        print("\nGenerated benchmark reports & training curves:")
        for fmt, path in report_files.items():
            print(f"  - [{fmt.upper()}]: {path}")

    sys.stdout = old_stdout
    dual_logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())


