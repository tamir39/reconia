"""Training pipeline for the Spatial Relational GNN model (Model 1).

Supports all 3 benchmark datasets (SROIE, CORD-v2, DocILE).
Profiles GPU VRAM, inference latency, token metrics, and entity metrics.

Usage:
    python -m app.training.train_gnn --dataset sroie --epochs 10 --device cuda
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        "token_metrics": token_metrics,
        "entity_metrics": entity_metrics,
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
    if "O" in vocabulary:
        # Downweight background 'O' tag so the model prioritizes identifying key entities
        weights[vocabulary["O"]] *= 0.25
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def main(argv: list[str] | None = None) -> int:
    if torch is None:
        print("ERROR: PyTorch is required to run train_gnn.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sroie", choices=sorted(ADAPTERS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-accum", type=int, default=8)
    parser.add_argument("--no-weighted-loss", action="store_true", help="disable class weighting")
    parser.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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

    print(f"=== Starting GNN Training (Model 1) ===")
    print(f"Dataset:    {args.dataset}")
    print(f"Device:     {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs:     {args.epochs}")
    print(f"Hidden Dim: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f"Cache Dir:  {args.cache_dir}")
    print(f"Report Dir: {args.report_dir}")

    # 1. Load Dataset
    adapter_cls = ADAPTERS[args.dataset]
    adapter: DatasetAdapter = adapter_cls()

    train_docs = list(adapter.iter_documents("train", limit=args.limit_train))
    val_docs = list(adapter.iter_documents("val", limit=args.limit_val))
    print(f"Loaded {len(train_docs)} train documents, {len(val_docs)} validation documents.")

    # 2. Vocabulary setup
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.cache_dir / f"{args.dataset}-vocabulary.json"
    if vocab_path.exists():
        vocabulary, fieldtypes = load_vocabulary(vocab_path)
        print(f"Loaded existing vocabulary: {len(vocabulary)} BIO labels from {vocab_path}")
    else:
        print("Building label vocabulary from full training split...")
        vocabulary, fieldtypes = build_vocabulary(adapter.iter_documents("train"))
        save_vocabulary(vocab_path, vocabulary, fieldtypes)
        print(f"Built & saved vocabulary: {len(vocabulary)} BIO labels to {vocab_path}")

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
    print(f"Total trainable parameters: {total_params:,}")

    # Loss & Optimizer
    if not args.no_weighted_loss:
        class_weights = compute_class_weights(train_docs, vocabulary, cache).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Enabled class-frequency weighted loss to balance entity extraction.")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = args.save_dir / f"gnn_{args.dataset}_best.pt"
    best_macro_f1 = -1.0
    best_val_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    print("\n" + "=" * 95)
    header = f"{'Epoch':<6} | {'TrainLoss':<10} | {'ValLoss':<10} | {'TokMicroF1':<11} | {'TokMacroF1':<11} | {'EntMicroF1':<11} | {'Lat(ms)':<8} | {'VRAM(MB)':<8}"
    print(header)
    print("=" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, cache, train_docs, optimizer, criterion, device, batch_accum=args.batch_accum
        )
        scheduler.step()

        val_metrics = evaluate_model(model, cache, val_docs, criterion, device, vocabulary)
        val_loss = val_metrics["val_loss"]
        tok_micro = val_metrics["token_metrics"]["micro_f1"]
        tok_macro = val_metrics["token_metrics"]["macro_f1"]
        ent_micro = val_metrics["entity_metrics"]["f1"]
        ent_macro = val_metrics["entity_metrics"].get("macro_f1", 0.0)
        lat = val_metrics["perf"]["latency_ms_per_doc"]
        vram = val_metrics["perf"]["peak_vram_mb"]

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "tok_micro_f1": float(tok_micro),
            "tok_macro_f1": float(tok_macro),
            "ent_micro_f1": float(ent_micro),
            "ent_macro_f1": float(ent_macro),
            "latency_ms": float(lat),
            "peak_vram_mb": float(vram),
        })

        print(
            f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {tok_micro:<11.4f} | "
            f"{tok_macro:<11.4f} | {ent_micro:<11.4f} | {lat:<8.1f} | {vram:<8.1f}"
        )

        if tok_macro > best_macro_f1:
            best_macro_f1 = tok_macro
            best_val_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": gnn_config,
                    "vocabulary": vocabulary,
                    "fieldtypes": fieldtypes,
                    "metrics": val_metrics,
                },
                best_checkpoint_path,
            )

    print("=" * 95)
    print(f"\nTraining completed. Best Token Macro-F1: {best_macro_f1:.4f}")
    print(f"Best model checkpoint saved to: {best_checkpoint_path}")

    # Export human-readable reports and charts
    metadata = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_params": total_params,
        "config": {
            "geo_dim": gnn_config.geo_dim,
            "text_dim": gnn_config.text_dim,
            "hidden_dim": gnn_config.hidden_dim,
            "num_layers": gnn_config.num_layers,
            "num_classes": gnn_config.num_classes,
            "dropout": gnn_config.dropout,
        },
        "batch_accum": args.batch_accum,
        "lr": args.lr,
        "seed": args.seed,
    }

    report_files = export_full_benchmark_report(
        output_dir=args.report_dir,
        model_name="GNN",
        dataset_name=args.dataset,
        history=history,
        final_val_metrics=best_val_metrics or val_metrics,
        metadata=metadata,
    )

    print("\nGenerated benchmark reports & comparison figures:")
    for fmt, path in report_files.items():
        print(f"  - [{fmt.upper()}]: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
