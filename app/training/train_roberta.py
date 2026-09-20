"""Training pipeline for RoBERTa with 2D Layout Positional Embedding (Model 2).

Fine-tunes RoBERTa-base augmented with 2D bounding box embeddings on invoice
datasets (SROIE, CORD-v2, DocILE).

Includes:
- Dual logging to console and timestamped file in reports/
- Early stopping & checkpoint resumption (best & latest)
- Gradient accumulation & clipping
- Memory & inference latency profiling
- Unified benchmark reporting (JSON, CSV, Markdown, PNG curves, comparative dashboard)

Usage:
    python -m app.training.train_roberta --dataset sroie --epochs 10 --batch-size 4 --lr 3e-5
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
from app.data.roberta_dataset import (
    RoBERTaInvoiceDataset,
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
)
from app.evaluation.report import generate_benchmark_summary
from app.graph.cache import build_vocabulary, load_vocabulary, save_vocabulary
from app.graph.labels import label_document
from app.models_dl.roberta import (
    RoBERTaWithLayoutConfig,
    RoBERTaWithLayoutForTokenClassification,
)

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache"
DEFAULT_CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"
DEFAULT_REPORTS = Path(__file__).resolve().parents[2] / "reports"

ADAPTERS: dict[str, type] = {
    "sroie": SroieAdapter,
    "cord": CordAdapter,
    "docile": DocileAdapter,
}


class DualLogger:
    """Duplicate stdout to both console and a log file."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except UnicodeEncodeError:
            self.terminal.write(message.encode("ascii", errors="replace").decode("ascii"))
            self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        try:
            self.log_file.close()
        except Exception:
            pass


def compute_class_weights(
    documents: list[Document],
    vocab: dict[str, int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse class frequency weights to balance rare entity classes."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for doc in documents:
        tags, _ = label_document(doc)
        for tag in tags:
            cid = vocab.get(tag, 0)
            if 0 <= cid < num_classes:
                counts[cid] += 1.0

    total_tokens = counts.sum()
    if total_tokens == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    # Balanced weighting: total / (num_classes * count)
    weights = np.zeros(num_classes, dtype=np.float64)
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total_tokens / (num_classes * counts[i])
        else:
            weights[i] = 1.0

    # Down-weight 'O' class (index 0) so model focuses on entities
    weights[0] = 0.2
    # Normalize
    weights = weights / weights.mean()
    # Clip extremes
    weights = np.clip(weights, 0.1, 10.0)
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

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

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

            preds = torch.argmax(logits, dim=-1)

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
            "latency_ms_per_doc": 0.0,
            "throughput_docs_per_sec": 0.0,
            "peak_vram_mb": 0.0,
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

    for doc in val_docs:
        encoded = encode_document_for_roberta(doc, tokenizer=tokenizer, label_vocabulary=vocab)
        inp_ids = encoded["input_ids"].unsqueeze(0).to(device)
        bb = encoded["bbox"].unsqueeze(0).to(device)
        att_m = encoded["attention_mask"].unsqueeze(0).to(device)
        lbls = encoded["labels"].cpu().numpy()

        with torch.no_grad():
            logits = model(inp_ids, bb, att_m)
            doc_preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        word_texts, _, word_gt_labels = extract_word_bboxes_and_labels(doc, vocab)
        word_gt_tags = [inv_vocab.get(lbl, "O") for lbl in word_gt_labels]
        word_pred_tags = ["O"] * len(word_texts)

        valid_indices = np.where(lbls != -100)[0]
        for i, tok_idx in enumerate(valid_indices):
            if i < len(word_texts):
                word_pred_tags[i] = inv_vocab.get(int(doc_preds[tok_idx]), "O")

        gt_ents = extract_entities_from_bio(word_gt_tags, texts=word_texts)
        pred_ents = extract_entities_from_bio(word_pred_tags, texts=word_texts)

        true_entities_batch.append(gt_ents)
        pred_entities_batch.append(pred_ents)

    entity_metrics = compute_entity_metrics(
        pred_entities_batch=pred_entities_batch,
        true_entities_batch=true_entities_batch,
        match_mode="exact",
    )

    return {
        "val_loss": avg_val_loss,
        "token_metrics": token_metrics,
        "entity_metrics": entity_metrics,
        "perf": perf_stats,
    }


def main(argv: list[str] | None = None) -> int:
    if torch is None:
        print("ERROR: PyTorch and transformers are required to run train_roberta.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sroie", choices=sorted(ADAPTERS), help="Dataset to train on (sroie, cord, docile)")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Peak learning rate for AdamW (typically 3e-5 for fine-tuning)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW regularizer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device (default: 4 for 4GB VRAM)")
    parser.add_argument("--batch-accum", type=int, default=4, help="Gradient accumulation steps (effective batch size = batch_size * batch_accum)")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping (<= 0 to disable)")
    parser.add_argument("--patience", "--early-stopping", dest="patience", type=int, default=5, help="Early stopping patience in epochs (0 to disable)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--monitor-metric", default="tok_macro", choices=["tok_macro", "ent_f1", "val_loss"], help="Metric to monitor for best checkpoint and early stopping")
    parser.add_argument("--no-weighted-loss", action="store_true", help="Disable class weighting in CrossEntropyLoss")
    parser.add_argument("--device", default="auto", help="Execution device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Path to precomputed feature/vocabulary cache")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_CHECKPOINTS, help="Directory to save model checkpoints")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS, help="Directory to save logs, metrics, and figures")
    parser.add_argument("--limit-train", type=int, default=None, help="Limit number of training documents (debugging)")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit number of validation documents (debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    # Enable automated dual logging
    args.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_file = args.report_dir / f"train_roberta_{args.dataset}_{timestamp_str}.log"
    dual_logger = DualLogger(log_file)
    old_stdout = sys.stdout
    sys.stdout = dual_logger

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== Starting RoBERTa Training (Model 2: RoBERTa-base + Layout Positional Embedding) ===")
    print(f"Dataset:     {args.dataset}")
    print(f"Device:      {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch Size:  {args.batch_size} (Gradient Accumulation: {args.batch_accum} -> Effective: {args.batch_size * args.batch_accum})")
    print(f"LR:          {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"Cache Dir:   {args.cache_dir}")
    print(f"Report Dir:  {args.report_dir}")

    # 1. Load dataset adapter
    adapter_cls = ADAPTERS[args.dataset]
    adapter: DatasetAdapter = adapter_cls()

    train_docs = list(adapter.iter_documents("train", limit=args.limit_train))
    val_docs = list(adapter.iter_documents("val", limit=args.limit_val))

    print(f"Loaded {len(train_docs)} train documents, {len(val_docs)} validation documents.")

    # 2. Vocabulary
    vocab_file = args.cache_dir / f"{args.dataset}-vocabulary.json"
    if vocab_file.exists():
        vocab, fieldtypes = load_vocabulary(vocab_file)
        print(f"Loaded existing vocabulary: {len(vocab)} BIO labels from {vocab_file}")
    else:
        full_train_docs = list(adapter.iter_documents("train"))
        vocab, fieldtypes = build_vocabulary(full_train_docs)
        save_vocabulary(vocab_file, vocab, fieldtypes)
        print(f"Built and saved vocabulary: {len(vocab)} BIO labels to {vocab_file}")

    inv_vocab = {idx: name for name, idx in vocab.items()}
    num_classes = len(vocab)

    # 3. Tokenizer & Dataset
    print("Loading RobertaTokenizerFast (roberta-base)...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    train_dataset = RoBERTaInvoiceDataset(train_docs, tokenizer=tokenizer, label_vocabulary=vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Model setup
    model_cfg = RoBERTaWithLayoutConfig(
        model_name_or_path="roberta-base",
        num_labels=num_classes,
        label2id=vocab,
        id2label=inv_vocab,
    )
    model = RoBERTaWithLayoutForTokenClassification(config=model_cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # Class weighting
    class_weights = None
    if not args.no_weighted_loss:
        class_weights = compute_class_weights(train_docs, vocab, num_classes, device)
        print("Enabled class-frequency weighted loss to balance entity extraction.")

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Early stopping & Resumption tracking
    start_epoch = 1
    best_monitor_val = float("-inf") if args.monitor_metric != "val_loss" else float("inf")
    patience_counter = 0
    train_history: list[dict[str, Any]] = []

    if args.resume and args.resume.exists():
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_monitor_val = ckpt.get("best_monitor_val", best_monitor_val)
        train_history = ckpt.get("train_history", [])
        print(f"Resumed at Epoch {start_epoch} with prior best {args.monitor_metric}: {best_monitor_val:.4f}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = args.save_dir / f"roberta_{args.dataset}_best.pt"
    latest_checkpoint_path = args.save_dir / f"roberta_{args.dataset}_latest.pt"

    profiler = InferenceProfiler(device=device)

    print("\n" + "=" * 95)
    print(f"{'Epoch':<6} | {'TrainLoss':<10} | {'ValLoss':<10} | {'TokMicroF1':<11} | {'TokMacroF1':<11} | {'EntMicroF1':<11} | {'Lat(ms)':<8} | {'VRAM(MB)':<8}")
    print("=" * 95)

    best_eval_metrics: dict[str, Any] = {}
    best_epoch_num = 1

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

            tok_micro = val_metrics["token_metrics"]["micro_f1"]
            tok_macro = val_metrics["token_metrics"]["macro_f1"]
            ent_micro = val_metrics["entity_metrics"]["f1"]
            ent_macro = val_metrics["entity_metrics"].get("macro_f1", 0.0)
            val_loss = val_metrics["val_loss"]
            lat_ms = val_metrics["perf"]["latency_ms_per_doc"]
            vram_mb = val_metrics["perf"]["peak_vram_mb"]

            print(
                f"{epoch:<6} | {avg_train_loss:<10.4f} | {val_loss:<10.4f} | "
                f"{tok_micro:<11.4f} | {tok_macro:<11.4f} | {ent_micro:<11.4f} | "
                f"{lat_ms:<8.1f} | {vram_mb:<8.1f}"
            )

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "tok_micro_f1": tok_micro,
                "tok_macro_f1": tok_macro,
                "ent_micro_f1": ent_micro,
                "ent_macro_f1": ent_macro,
                "latency_ms": lat_ms,
                "peak_vram_mb": vram_mb,
            }
            train_history.append(epoch_record)

            # Determine improvement
            current_monitor = tok_macro
            if args.monitor_metric == "ent_f1":
                current_monitor = ent_micro
            elif args.monitor_metric == "val_loss":
                current_monitor = val_loss

            is_improvement = (
                current_monitor > best_monitor_val
                if args.monitor_metric != "val_loss"
                else current_monitor < best_monitor_val
            )

            # Save latest checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_monitor_val": best_monitor_val,
                    "train_history": train_history,
                },
                latest_checkpoint_path,
            )

            if is_improvement:
                best_monitor_val = current_monitor
                patience_counter = 0
                best_epoch_num = epoch
                best_eval_metrics = val_metrics
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": model_cfg.to_dict(),
                        "best_monitor_val": best_monitor_val,
                    },
                    best_checkpoint_path,
                )
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"\n[Early Stopping Triggered] No improvement in {args.monitor_metric} for {args.patience} epochs.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Finalizing reports from completed epochs...")

    print("=" * 95)
    print(f"\nTraining completed. Best {args.monitor_metric}: {best_monitor_val:.4f} (at Epoch {best_epoch_num})")
    print(f"Best model checkpoint saved to: {best_checkpoint_path}")
    print(f"Latest checkpoint saved to:       {latest_checkpoint_path}")

    # 5. Export comprehensive benchmark reports and curves
    if (best_eval_metrics or val_metrics) and train_history:
        metadata = {
            "device": f"{device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "epochs_trained": len(train_history),
            "best_epoch": best_epoch_num,
            "batch_size": args.batch_size,
            "batch_accum": args.batch_accum,
            "lr": args.lr,
            "seed": args.seed,
        }

        report_files = export_full_benchmark_report(
            output_dir=args.report_dir,
            model_name="RoBERTa",
            dataset_name=args.dataset,
            history=train_history,
            final_val_metrics=best_eval_metrics or val_metrics,
            metadata=metadata,
        )

        print("\nGenerated benchmark reports & comparison figures:")
        for fmt, path in report_files.items():
            print(f"  - [{fmt.upper()}]: {path}")



    sys.stdout = old_stdout
    dual_logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
