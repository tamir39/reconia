"""Training pipeline for RoBERTa with 2D Layout Positional Embedding (Model 2).

Fine-tunes RoBERTa-base on invoice datasets (SROIE, CORD-v2, DocILE).

Usage:
    python -m app.training.train_roberta --dataset sroie --epochs 20
"""

from __future__ import annotations

import argparse
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
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
except ImportError:
    torch = None

from app.data import CordAdapter, DocileAdapter, SroieAdapter
from app.data.paths import WORKSPACE_ROOT
from app.data.portable import load_adapter
from app.data.roberta_dataset import (
    RoBERTaInvoiceDataset,
    WINDOW_POLICY,
    encode_document_for_roberta,
    extract_word_bboxes_and_labels,
)
from app.data.schema import DatasetAdapter, Document
from app.evaluation import (
    InferenceProfiler,
    compute_entity_metrics,
    compute_token_metrics,
    export_full_benchmark_report,
    extract_entities_from_bio,
    save_training_history_csv,
)
from app.graph.cache import build_vocabulary, load_vocabulary, save_vocabulary
from app.graph.labels import label_document
from app.models_dl.roberta import (
    RoBERTaWithLayoutConfig,
    RoBERTaWithLayoutForTokenClassification,
    get_default_roberta_model_path,
)

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache"
DEFAULT_CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"
DEFAULT_REPORTS = Path(__file__).resolve().parents[2] / "reports"

ADAPTERS: dict[str, type] = {
    "sroie": SroieAdapter,
    "cord": CordAdapter,
    "docile": DocileAdapter,
}


from app.training.utils import DualLogger, str2bool


def compute_class_weights(
    documents: list[Document],
    vocab: dict[str, int],
    num_classes: int,
    device: torch.device,
    power: float = 0.5,
) -> torch.Tensor:
    """Compute smoothed inverse class frequency weights to balance rare entity classes."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for doc in documents:
        tags, _ = label_document(doc)
        for tag in tags:
            cid = vocab.get(tag, 0)
            if 0 <= cid < num_classes:
                counts[cid] += 1.0

    mask = counts > 0
    weights = np.ones(num_classes, dtype=np.float64)
    if mask.any():
        weights[mask] = 1.0 / (counts[mask] ** power)
        weights[mask] = weights[mask] / weights[mask].mean()
        weights = np.clip(weights, 0.2, 5.0)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_roberta_model(
    model: RoBERTaWithLayoutForTokenClassification,
    val_docs: list[Document],
    tokenizer: Any,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    device: torch.device,
    batch_size: int = 4,
    profiler: InferenceProfiler | None = None,
) -> tuple[dict[str, float], dict[str, Any], list[int], list[int]]:
    """Evaluate RoBERTa model on validation documents."""
    model.eval()
    val_dataset = RoBERTaInvoiceDataset(
        val_docs,
        tokenizer=tokenizer,
        label_vocabulary=vocab,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    all_preds: list[int] = []
    all_targets: list[int] = []
    total_val_loss = 0.0
    val_steps = 0
    document_predictions = [["O"] * len(doc.words) for doc in val_docs]

    if profiler:
        profiler.start()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_val_loss += loss.item()
            val_steps += 1

            preds = torch.argmax(logits, dim=-1).cpu()
            for row, doc_index in enumerate(batch["document_index"].tolist()):
                for position, word_index in enumerate(batch["word_indices"][row].tolist()):
                    if word_index >= 0:
                        document_predictions[doc_index][word_index] = inv_vocab.get(int(preds[row, position]), "O")

            flat_labels = labels.view(-1).cpu().numpy()
            flat_preds = preds.view(-1).cpu().numpy()
            valid_mask = flat_labels != -100

            all_preds.extend(flat_preds[valid_mask].tolist())
            all_targets.extend(flat_labels[valid_mask].tolist())

    perf_stats = (
        profiler.stop(num_documents=len(val_docs))
        if profiler
        else {
            "elapsed_sec": 0.0,
            "latency": 0.0,
            "throughput": 0.0,
        }
    )

    avg_val_loss = total_val_loss / max(1, val_steps)

    # 1. Token metrics
    token_metrics = compute_token_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        label_names=inv_vocab,
        ignore_labels={"O"},
    )

    # 2. Entity-level metrics
    pred_entities_batch: list[list[EntitySpan]] = []
    true_entities_batch: list[list[EntitySpan]] = []

    for doc_index, doc in enumerate(val_docs):
        word_texts = [word.text for word in doc.words]
        word_gt_tags, _ = label_document(doc)
        word_pred_tags = document_predictions[doc_index]

        pages = [word.page for word in doc.words]
        gt_ents = extract_entities_from_bio(word_gt_tags, texts=word_texts, pages=pages)
        pred_ents = extract_entities_from_bio(word_pred_tags, texts=word_texts, pages=pages)

        true_entities_batch.append(gt_ents)
        pred_entities_batch.append(pred_ents)

    entity_metrics = compute_entity_metrics(
        pred_entities_batch=pred_entities_batch,
        true_entities_batch=true_entities_batch,
        match_mode="exact",
    )

    return {
        "val_loss": avg_val_loss,
        "precision": float(entity_metrics["precision"]),
        "recall": float(entity_metrics["recall"]),
        "f1": float(entity_metrics["f1"]),
        "entity_metrics": entity_metrics,
        "token_metrics": token_metrics,
        "perf": perf_stats,
    }


def main(argv: list[str] | None = None) -> int:
    if torch is None:
        print("ERROR: PyTorch and transformers are required to run train_roberta.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "sroie", "cord", "docile"],
        help="Dataset to train on: 'all' (default: sroie cord docile), 'sroie', 'cord', 'docile'",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate for RoBERTa backbone (default: 5e-5)")
    parser.add_argument("--head-lr", type=float, default=2e-4, help="Learning rate for layout embeddings and classification head (default: 2e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW regularizer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device (default: 4 for 4GB VRAM)")
    parser.add_argument("--batch-accum", type=int, default=4, help="Gradient accumulation steps (effective batch size = batch_size * batch_accum)")
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
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Path to precomputed feature/vocabulary cache")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_CHECKPOINTS, help="Directory to save model checkpoints")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS, help="Directory to save logs, metrics, and figures")
    parser.add_argument("--limit-train", type=int, default=None, help="Limit number of training documents (debugging)")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit number of validation documents (debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    target_datasets = ["sroie", "cord", "docile"] if args.dataset == "all" else [args.dataset]
    for current_dataset in target_datasets:
        train_single_roberta_dataset(args, current_dataset, device)
    return 0


def train_single_roberta_dataset(args: argparse.Namespace, dataset_name: str, device: torch.device) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.report_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_roberta_{dataset_name}_{timestamp_str}.log"
    dual_logger = DualLogger(log_file)
    old_stdout = sys.stdout
    sys.stdout = dual_logger

    # 1. Load dataset adapter
    adapter_cls = ADAPTERS[dataset_name]
    adapter: DatasetAdapter = load_adapter(dataset_name, adapter_cls)

    train_docs = list(adapter.iter_documents("train", limit=args.limit_train))
    val_docs = list(adapter.iter_documents("val", limit=args.limit_val))

    # 2. Vocabulary
    vocab_file = args.cache_dir / f"{dataset_name}-vocabulary.json"
    if vocab_file.exists():
        vocab, fieldtypes = load_vocabulary(vocab_file)
    else:
        full_train_docs = list(adapter.iter_documents("train"))
        vocab, fieldtypes = build_vocabulary(full_train_docs)
        save_vocabulary(vocab_file, vocab, fieldtypes)

    inv_vocab = {idx: name for name, idx in vocab.items()}
    num_classes = len(vocab)

    # 3. Tokenizer & Dataset
    model_source = get_default_roberta_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_source, add_prefix_space=True)
    tokenizer.save_pretrained(args.save_dir / "tokenizer")

    train_dataset = RoBERTaInvoiceDataset(train_docs, tokenizer=tokenizer, label_vocabulary=vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Model setup
    model_cfg = RoBERTaWithLayoutConfig(
        model_name_or_path=model_source,
        num_labels=num_classes,
        label2id=vocab,
        id2label=inv_vocab,
    )
    model = RoBERTaWithLayoutForTokenClassification(config=model_cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Class weighting (optional frequency-weighted loss to balance entity extraction)
    class_weights = None
    if args.weighted_loss:
        class_weights = compute_class_weights(train_docs, vocab, num_classes, device)

    # Optimizer & Scheduler with differential learning rates (backbone vs new heads)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    backbone_decay = [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay)]
    backbone_no_decay = [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay)]
    head_params = (
        [p for n, p in model.layout_embeddings.named_parameters()]
        + [p for n, p in model.classifier.named_parameters()]
    )

    optimizer_grouped_parameters = [
        {
            "params": backbone_decay,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": backbone_no_decay,
            "lr": args.lr,
            "weight_decay": 0.0,
        },
        {
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Early stopping & Resumption tracking
    start_epoch = 1
    best_monitor_val = float("-inf") if args.monitor_metric != "val_loss" else float("inf")
    patience_counter = 0
    train_history: list[dict[str, Any]] = []
    best_epoch_num = 1
    best_eval_metrics: dict[str, Any] = {}

    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("vocabulary") != vocab or ckpt.get("window_policy") != WINDOW_POLICY:
            raise ValueError("Resume requires the same vocabulary and full-document window policy")
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_monitor_val = ckpt.get("best_monitor_val", best_monitor_val)
        train_history = ckpt.get("train_history", [])
        best_epoch_num = ckpt.get("best_epoch", 1)
        best_eval_metrics = ckpt.get("best_metrics", {})
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resumed at Epoch {start_epoch} with prior best {args.monitor_metric}: {best_monitor_val:.4f}")

    is_debug = bool(args.limit_train is not None or args.limit_val is not None)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = args.save_dir / f"roberta_{dataset_name}_best.pt"
    latest_checkpoint_path = args.save_dir / f"roberta_{dataset_name}_latest.pt"

    metrics_dir = args.report_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = metrics_dir / f"roberta_{dataset_name.lower()}_history.csv"

    profiler = InferenceProfiler(device=device)

    gpu_desc = f"{torch.cuda.get_device_name(device)}" if device.type == "cuda" else "CPU"
    loss_mode = "CrossEntropy (Weighted: True - Inverse class frequency)" if args.weighted_loss else "CrossEntropy (Weighted: False - Standard)"
    eff_batch = args.batch_size * args.batch_accum
    patience_desc = f"{args.patience} epochs (monitoring '{args.monitor_metric}')" if args.patience > 0 else "Disabled"
    fields_desc = f"{len(fieldtypes)} entity types ({', '.join(sorted(fieldtypes))})" if fieldtypes else "N/A"

    print(f"\n{'=' * 80}")
    print(f"{'RECONIA EXPERIMENT SETUP: ROBERTA-BASE':^80}")
    print(f"{'=' * 80}")
    print(f"[Hardware & Runtime]")
    print(f"  Device              : {str(device).upper()} ({gpu_desc})")
    print(f"  Random Seed         : {args.seed}")
    print(f"\n[Dataset & Labels]")
    print(f"  Target Dataset      : {dataset_name.upper()} (Train: {len(train_docs)} docs | Val: {len(val_docs)} docs)")
    print(f"  Vocabulary          : {len(vocab)} BIO tags across {fields_desc}")
    print(f"\n[Model Architecture]")
    print(f"  Backbone            : roberta-base (12 layers, 768 hidden, 12 heads)")
    print(f"  Layout Module       : 2D Spatial Positional Embeddings (x0, y0, x1, y1)")
    print(f"  Parameters          : {trainable_params:,} trainable parameters")
    print(f"\n[Training Hyperparameters]")
    print(f"  Batch Strategy      : Batch size: {args.batch_size} x Accum: {args.batch_accum}  ==>  Effective Batch Size: {eff_batch}")
    print(f"  Learning Rates      : Backbone LR: {args.lr:.1e} | Head LR: {args.head_lr:.1e}")
    print(f"  Optimization        : AdamW (Weight Decay: {args.weight_decay} | Clip Norm: {args.clip_grad_norm})")
    print(f"  Schedule & EarlyStop: {args.epochs} max epochs (Patience: {patience_desc})")
    print(f"  Loss Criterion      : {loss_mode}")
    print(f"\n[Artifacts & Outputs]")
    print(f"  Best Checkpoint     : {best_checkpoint_path}")
    print(f"  Reports & Charts    : {args.report_dir}")
    print(f"{'=' * 80}")

    print("\n" + "=" * 80)
    print(f"{'Epoch':<6} | {'TrainLoss':<10} | {'ValLoss':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Latency(ms)':<11}")
    print("=" * 80)

    val_metrics: dict[str, Any] = {}

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_train_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss, _ = model(
                    input_ids=input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=class_weights,
                )

                loss = loss / args.batch_accum
                loss.backward()
                total_train_loss += loss.item() * args.batch_accum

                if (step + 1) % args.batch_accum == 0 or (step + 1) == len(train_loader):
                    if args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()
            avg_train_loss = total_train_loss / max(1, len(train_loader))

            # Validation
            val_metrics = evaluate_roberta_model(
                model=model,
                val_docs=val_docs,
                tokenizer=tokenizer,
                vocab=vocab,
                inv_vocab=inv_vocab,
                device=device,
                batch_size=args.batch_size,
                profiler=profiler,
            )

            val_loss = val_metrics["val_loss"]
            precision = val_metrics["precision"]
            recall = val_metrics["recall"]
            f1 = val_metrics["f1"]
            lat = val_metrics["perf"]["latency"]

            print(
                f"{epoch:<6} | {avg_train_loss:<10.4f} | {val_loss:<10.4f} | "
                f"{precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {lat:<11.1f}"
            )

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "latency": lat,
            }
            train_history.append(epoch_record)

            # Real-time persistence of training history (full training only)
            if not is_debug:
                save_training_history_csv(history_csv_path, train_history)

            # Determine improvement
            current_monitor = f1 if args.monitor_metric == "f1" else val_loss

            is_improvement = (
                current_monitor > best_monitor_val
                if args.monitor_metric != "val_loss"
                else current_monitor < best_monitor_val
            )

            # Unified checkpoint state
            checkpoint_state = {
                "window_policy": WINDOW_POLICY,
                "best_metrics": val_metrics if is_improvement else best_eval_metrics,
                "patience_counter": 0 if is_improvement else patience_counter + 1,
                "epoch": epoch,
                "best_epoch": epoch if is_improvement else best_epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": model_cfg.to_dict(),
                "backbone_config": model.roberta.config.to_dict(),
                "vocabulary": vocab,
                "best_score": current_monitor if is_improvement else best_monitor_val,
                "best_monitor_val": current_monitor if is_improvement else best_monitor_val,
                "history": train_history,
                "train_history": train_history,
                "metrics": val_metrics,
                "monitor_metric": args.monitor_metric,
            }

            # Always save latest checkpoint
            torch.save(checkpoint_state, latest_checkpoint_path)

            if is_improvement:
                best_monitor_val = current_monitor
                patience_counter = 0
                best_epoch_num = epoch
                best_eval_metrics = val_metrics
                torch.save(checkpoint_state, best_checkpoint_path)
            else:
                if best_monitor_val > 0.0 or args.monitor_metric == "val_loss":
                    patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\n[Early Stopping] No improvement in '{args.monitor_metric}' for {args.patience} consecutive epochs. Stopping at epoch {epoch}.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Finalizing reports from completed epochs...")

    print("=" * 87)
    print(f"\nTraining completed. Best {args.monitor_metric}: {best_monitor_val:.4f} (at Epoch {best_epoch_num})")
    print(f"Best model checkpoint saved to: {best_checkpoint_path}")
    print(f"Latest checkpoint saved to:       {latest_checkpoint_path}")

    # 5. Automatic Test Set Evaluation & Prediction Export on best checkpoint (Full Training only)
    test_metrics: dict[str, Any] | None = None
    if is_debug:
        print(f"\n[Debug Mode Detected] Skipping TEST set evaluation, prediction export, and benchmark reporting.")
    elif "test" in adapter.splits() and getattr(adapter, "has_ground_truth", lambda s: True)("test"):
        print(f"\n=== Evaluating Best Checkpoint & Exporting Predictions on TEST Set ({dataset_name.upper()}) ===")
        pred_dir = args.report_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"predictions_roberta_{dataset_name.lower()}_test.json"

        from app.evaluation.predict import run_prediction_pipeline

        test_metrics = run_prediction_pipeline(
            model_path=best_checkpoint_path,
            dataset_name=dataset_name,
            split="test",
            output_json=pred_file,
            device_str=str(device),
            cache_dir=args.cache_dir,
            model_type="roberta",
        )
        p_t = test_metrics["precision"] * 100
        r_t = test_metrics["recall"] * 100
        f1_t = test_metrics["f1"] * 100
        lat_t = test_metrics.get("latency", test_metrics.get("perf", {}).get("latency", 0.0))
        print(f"TEST Results (Best Checkpoint - Epoch {best_epoch_num}):")
        print(f"  Precision: {p_t:.2f}% | Recall: {r_t:.2f}% | F1: {f1_t:.2f}% | Latency: {lat_t:.1f} ms")
        print(f"  Predictions saved to: {pred_file}")

    # 6. Export comprehensive benchmark reports (Full Training only)
    if not is_debug and (best_eval_metrics or val_metrics) and train_history:
        metadata = {
            "window_policy": WINDOW_POLICY,
            "train_windows": len(train_dataset),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "train_documents": len(train_docs),
            "val_documents": len(val_docs),
            "weighted_loss": args.weighted_loss,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "epochs_trained": len(train_history),
            "best_epoch": best_epoch_num,
            "batch_size": args.batch_size,
            "batch_accum": args.batch_accum,
            "effective_batch_size": args.batch_size * args.batch_accum,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "patience": args.patience,
            "monitor_metric": args.monitor_metric,
            "clip_grad_norm": args.clip_grad_norm,
            "resumed_from": str(args.resume) if args.resume else None,
            "seed": args.seed,
        }

        report_files = export_full_benchmark_report(
            output_dir=args.report_dir,
            model_name="RoBERTa",
            dataset_name=dataset_name,
            history=train_history,
            final_val_metrics=best_eval_metrics or val_metrics,
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
