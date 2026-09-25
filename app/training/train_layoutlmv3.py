"""Fine-tune LayoutLMv3 on verified page images and existing OCR splits."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from app.data import SroieAdapter, CordAdapter, DocileAdapter
from app.data.portable import load_adapter
from app.data.layoutlmv3_dataset import LayoutLMv3InvoiceDataset, WINDOW_POLICY
from app.evaluation.metrics import InferenceProfiler, compute_entity_metrics, extract_entities_from_bio
from app.evaluation.report import export_full_benchmark_report, save_training_history_csv
from app.evaluation.predict import run_prediction_pipeline
from app.graph.cache import build_vocabulary
from app.graph.labels import label_document
from app.models_dl.layoutlmv3 import BACKBONE_REPO, BACKBONE_REVISION
from app.training.train_roberta import compute_class_weights
from app.training.utils import DualLogger

INPUT_KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values")


def evaluate(model, dataset, vocabulary, device, batch_size):
    model.eval()
    inverse = {i: tag for tag, i in vocabulary.items()}
    tags = [["O"] * len(doc.words) for doc in dataset.documents]
    total_loss, tokens = 0.0, 0
    profiler = InferenceProfiler(device=device)
    profiler.start()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            logits = model(**{k: batch[k].to(device) for k in INPUT_KEYS}).logits
            labels = batch["labels"].to(device)
            total_loss += float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"))
            tokens += int((labels != -100).sum())
            predictions = logits.argmax(-1).cpu()
            for row, doc_index in enumerate(batch["document_index"].tolist()):
                for col, word_index in enumerate(batch["word_indices"][row].tolist()):
                    if word_index >= 0:
                        tags[doc_index][word_index] = inverse[int(predictions[row, col])]
    perf = profiler.stop(num_documents=len(dataset.documents))
    predicted, expected = [], []
    for doc, doc_tags in zip(dataset.documents, tags):
        kwargs = dict(texts=[w.text for w in doc.words], pages=[w.page for w in doc.words])
        predicted.append(extract_entities_from_bio(doc_tags, **kwargs))
        expected.append(extract_entities_from_bio(label_document(doc)[0], **kwargs))
    metrics = compute_entity_metrics(predicted, expected, match_mode="exact")
    return {k: float(metrics[k]) for k in ("precision", "recall", "f1")} | {
        "entity_metrics": metrics, "val_loss": total_loss / max(tokens, 1),
        "latency": perf["latency"], "perf": perf,
    }


def train(args):
    if min(args.epochs, args.batch_size, args.batch_accum) < 1 or args.patience < 0:
        raise ValueError("Epochs and batch sizes must be positive; patience must be nonnegative")
    if any(n is not None and n < 1 for n in (args.limit_train, args.limit_val)):
        raise ValueError("Document limits must be positive")
    if any(args.save_dir.glob("*.pt")):
        raise ValueError("Checkpoint directory already used. Choose a new run directory.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    adapter = load_adapter(args.dataset, {"sroie": SroieAdapter, "cord": CordAdapter, "docile": DocileAdapter}[args.dataset])
    split_ids = {s: adapter.document_ids(s) for s in ("train", "val", "test")}
    if any(len(ids) != len(set(ids)) for ids in split_ids.values()):
        raise ValueError("Duplicate IDs within split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(split_ids[a]) & set(split_ids[b]):
            raise ValueError("Dataset splits overlap")
    all_train = list(adapter.iter_documents("train"))
    vocabulary, _ = build_vocabulary(all_train)
    train_docs = all_train[:args.limit_train] if args.limit_train else all_train
    val_docs = list(adapter.iter_documents("val", limit=args.limit_val))
    source = args.model_path or BACKBONE_REPO
    kwargs = {} if args.model_path else {"revision": BACKBONE_REVISION}
    processor = AutoProcessor.from_pretrained(source, apply_ocr=False, **kwargs)
    train_data = LayoutLMv3InvoiceDataset(train_docs, processor, vocabulary, adapter.get_image)
    val_data = LayoutLMv3InvoiceDataset(val_docs, processor, vocabulary, adapter.get_image)
    if not len(train_data) or not len(val_data):
        raise ValueError("Training and validation require OCR words and page images")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.save_dir / "processor")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        source, num_labels=len(vocabulary), label2id=vocabulary,
        id2label={i: tag for tag, i in vocabulary.items()}, **kwargs,
    ).to(device)
    groups = []
    for head in (False, True):
        for decay in (False, True):
            params = [p for name, p in model.named_parameters()
                      if name.startswith("classifier.") == head
                      and (not any(x in name for x in ("bias", "LayerNorm.weight"))) == decay]
            if params:
                groups.append(dict(params=params, lr=args.head_lr if head else args.lr,
                                   weight_decay=0.01 if decay else 0.0))
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    weights = compute_class_weights(train_docs, vocabulary, len(vocabulary), device)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    history, best_metrics = [], {}
    best_f1, best_epoch, patience = -1.0, 0, 0
    best_path = args.save_dir / f"layoutlmv3_{args.dataset}_best.pt"
    latest_path = args.save_dir / f"layoutlmv3_{args.dataset}_latest.pt"
    print(f"Target Dataset: {args.dataset} (Train: {len(train_docs)} docs | Val: {len(val_docs)} docs)", flush=True)
    print(f"Windows: {len(train_data)} train / {len(val_data)} val | Device: {device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            logits = model(**{k: batch[k].to(device) for k in INPUT_KEYS}).logits
            loss = F.cross_entropy(logits.flatten(0, 1), batch["labels"].to(device).flatten(),
                                   weight=weights, ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            # Normalize the last accumulation group by its actual number of batches.
            group_start = step // args.batch_accum * args.batch_accum
            group_size = min(args.batch_accum, len(loader) - group_start)
            (loss / group_size).backward()
            epoch_loss += float(loss.detach())
            if (step + 1) % args.batch_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        metrics = evaluate(model, val_data, vocabulary, device, args.batch_size)
        row = {"epoch": epoch, "train_loss": epoch_loss / len(loader),
               **{k: metrics[k] for k in ("val_loss", "precision", "recall", "f1", "latency")}}
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = metrics["f1"] > best_f1
        if improved:
            best_f1, best_epoch, best_metrics, patience = metrics["f1"], epoch, metrics, 0
        else:
            patience += 1
        checkpoint = dict(model_type="layoutlmv3", window_policy=WINDOW_POLICY,
            config=model.config.to_dict(), model_state_dict=model.state_dict(), vocabulary=vocabulary,
            epoch=epoch, best_epoch=best_epoch, best_metrics=best_metrics, history=history,
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            patience_counter=patience, seed=args.seed, backbone_repo=BACKBONE_REPO,
            backbone_revision=BACKBONE_REVISION, image_input="original-page-rgb", ocr_input="dataset-provided")
        torch.save(checkpoint, latest_path)
        if improved:
            torch.save(checkpoint, best_path)
        save_training_history_csv(args.report_dir / f"metrics/layoutlmv3_{args.dataset}_history.csv", history)
        if args.patience and patience >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    smoke = args.limit_train is not None or args.limit_val is not None
    metadata = dict(window_policy=WINDOW_POLICY, train_documents=len(train_docs), val_documents=len(val_docs),
        train_windows=len(train_data), best_epoch=best_epoch, epochs_trained=len(history),
        seed=args.seed, device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        total_params=sum(p.numel() for p in model.parameters()), batch_size=args.batch_size,
        batch_accum=args.batch_accum, lr=args.lr, head_lr=args.head_lr, patience=args.patience,
        backbone_repo=BACKBONE_REPO, backbone_revision=BACKBONE_REVISION,
        image_input="original-page-rgb", ocr_input="dataset-provided")
    if hasattr(adapter, "_image_store"):
        metadata.update(image_manifest_sha256=adapter._image_store.manifest_sha256,
                        image_render_policy=adapter._image_store.manifest["policy"])
    del checkpoint, model, optimizer, scheduler, groups, params, train_data, val_data, loader, logits, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not smoke:
        test = run_prediction_pipeline(best_path, args.dataset, model_type="layoutlmv3", split="test",
            output_json=args.report_dir / f"predictions/predictions_layoutlmv3_{args.dataset}_test.json",
            device_str=str(device))
        export_full_benchmark_report(output_dir=args.report_dir, model_name="LayoutLMv3",
            dataset_name=args.dataset, history=history, final_val_metrics=best_metrics,
            test_metrics=test, metadata=metadata)
    print(f"Training completed. Best F1: {best_f1:.4f} at epoch {best_epoch}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sroie", "cord", "docile"], required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    args = parser.parse_args(argv)
    logger = DualLogger(args.report_dir / "logs" / f"train_layoutlmv3_{args.dataset}.log")
    original = sys.stdout
    try:
        sys.stdout = logger
        train(args)
    finally:
        sys.stdout = original
        logger.close()


if __name__ == "__main__":
    main()
