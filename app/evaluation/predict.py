"""Model inference and evaluation pipeline for invoice extraction.

Supports GNN, RoBERTa, and future model architectures.

Capabilities:
  1. Computes entity-level Precision, Recall, F1, and Latency when ground-truth is available.
  2. Exports predictions in benchmark format (reports/predictions/predictions_<model>_<dataset>_<split>.json).
  3. Supports single model evaluation or batch evaluation (--all).

Usage:
    python -m app.evaluation.predict --model gnn --dataset sroie --split test
    python -m app.evaluation.predict --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, RobertaConfig, RobertaModel
except ImportError:
    torch = None
    nn = object
    F = None

from app.data import CordAdapter, DocileAdapter, SroieAdapter
from app.data.paths import WORKSPACE_ROOT
from app.data.portable import load_adapter
from app.data.roberta_dataset import encode_document_windows, extract_word_bboxes_and_labels, WINDOW_POLICY
from app.data.schema import DatasetAdapter, Document
from app.evaluation.values import compute_value_metrics
from app.evaluation.metrics import (
    EntitySpan,
    InferenceProfiler,
    compute_entity_metrics,
    compute_token_metrics,
    extract_entities_from_bio,
)
from app.graph.builder import build_graph
from app.graph.cache import load_vocabulary
from app.graph.edges import compute_spatial_edge_features
from app.graph.labels import label_document
from app.models_dl.gnn import InvoiceGNN, InvoiceGNNConfig, validate_gnn_checkpoint
from app.models_dl.roberta import (
    RoBERTaWithLayoutConfig,
    RoBERTaWithLayoutForTokenClassification,
)

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "sroie": SroieAdapter,
    "cord": CordAdapter,
    "docile": DocileAdapter,
}


def span_to_bbox(doc: Document, span: EntitySpan) -> list[float]:
    """Compute the union bounding box [x0, y0, x1, y1] for an EntitySpan in normalized coordinates [0.0, 1.0]."""
    words = doc.words[span.start_token : span.end_token]
    if not words:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = min(w.bbox.x0 for w in words)
    y0 = min(w.bbox.y0 for w in words)
    x1 = max(w.bbox.x1 for w in words)
    y1 = max(w.bbox.y1 for w in words)
    return [float(x0), float(y0), float(x1), float(y1)]


def spans_to_docile_predictions(
    doc: Document,
    spans: list[EntitySpan],
    confidences: list[float],
) -> list[dict[str, Any]]:
    """Convert extracted EntitySpans and token confidences into DocILE prediction records."""
    docile_predictions: list[dict[str, Any]] = []
    for span in spans:
        span_words = doc.words[span.start_token : span.end_token]
        page = span_words[0].page if span_words else 0
        box = span_to_bbox(doc, span)
        scores = confidences[span.start_token : span.end_token]
        mean_score = float(np.mean(scores)) if scores else 1.0

        pred_entry = {
            "page": page,
            "bbox": box,
            "fieldtype": span.entity_type,
            "score": round(mean_score, 4),
            "text": span.text or "",
        }
        docile_predictions.append(pred_entry)
    return docile_predictions


def predict_document_roberta(
    model: RoBERTaWithLayoutForTokenClassification,
    doc: Document,
    tokenizer: Any,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    device: torch.device,
) -> tuple[list[EntitySpan], list[dict[str, Any]]]:
    """Run RoBERTa inference on a single Document and return EntitySpans and DocILE formatted predictions."""
    word_texts, word_boxes, _ = extract_word_bboxes_and_labels(doc, vocab)
    if not word_texts:
        return [], []

    windows = encode_document_windows(
        document=doc,
        tokenizer=tokenizer,
        label_vocabulary=vocab,
        max_seq_length=512,
    )

    word_pred_tags = ["O"] * len(word_texts)
    word_confidences = [1.0] * len(word_texts)
    for encoded in windows:
        with torch.no_grad():
            out = model(input_ids=encoded["input_ids"].unsqueeze(0).to(device),
                        bbox=encoded["bbox"].unsqueeze(0).to(device),
                        attention_mask=encoded["attention_mask"].unsqueeze(0).to(device))
            logits = out[1] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            doc_preds = np.argmax(probs, axis=-1)
        for tok_idx, word_idx in enumerate(encoded["word_indices"].tolist()):
            if word_idx >= 0:
                pred_id = int(doc_preds[tok_idx])
                word_pred_tags[word_idx] = inv_vocab.get(pred_id, "O")
                word_confidences[word_idx] = float(probs[tok_idx, pred_id])

    pred_spans = extract_entities_from_bio(
        word_pred_tags, texts=word_texts, pages=[word.page for word in doc.words]
    )
    docile_predictions = spans_to_docile_predictions(doc, pred_spans, word_confidences)
    return pred_spans, docile_predictions


def predict_document_gnn(
    model: nn.Module,
    doc: Document,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    device: torch.device,
) -> tuple[list[EntitySpan], list[dict[str, Any]]]:
    """Run GNN inference on a single Document and return EntitySpans and DocILE formatted predictions."""
    if not doc.words:
        return [], []

    graph = build_graph(doc, vocab)
    if graph.num_nodes == 0:
        return [], []

    u_edge_index, u_edge_type = graph.to_undirected()
    spatial_attr = compute_spatial_edge_features(graph.features, u_edge_index)

    geo_tensor = torch.tensor(graph.features, dtype=torch.float32, device=device)
    edge_index_tensor = torch.tensor(u_edge_index, dtype=torch.long, device=device)
    edge_type_tensor = torch.tensor(u_edge_type, dtype=torch.long, device=device)
    edge_attr_tensor = torch.tensor(spatial_attr, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(geo_tensor, graph.texts, edge_index_tensor, edge_type_tensor, edge_attr_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    pred_tags = [inv_vocab.get(int(p), "O") for p in preds]
    confidences = [float(probs[i, preds[i]]) for i in range(len(preds))]

    pred_spans = extract_entities_from_bio(pred_tags, texts=graph.texts, pages=graph.pages)
    docile_predictions = spans_to_docile_predictions(doc, pred_spans, confidences)
    return pred_spans, docile_predictions


def predict_document_layoutlmv3(model, doc, processor, vocab, inv_vocab, device, image_loader):
    from app.data.layoutlmv3_dataset import encode_layoutlmv3_windows

    tags, confidences = ["O"] * len(doc.words), [1.0] * len(doc.words)
    for window in encode_layoutlmv3_windows(doc, processor, vocab, image_loader):
        inputs = {k: window[k].unsqueeze(0).to(device)
                  for k in ("input_ids", "attention_mask", "bbox", "pixel_values")}
        with torch.no_grad():
            probabilities = model(**inputs).logits.softmax(-1)[0].cpu()
        for position, index in enumerate(window["word_indices"].tolist()):
            if index >= 0:
                confidence, label = probabilities[position].max(-1)
                tags[index] = inv_vocab[int(label)]
                confidences[index] = float(confidence)
    spans = extract_entities_from_bio(tags, texts=[w.text for w in doc.words],
                                      pages=[w.page for w in doc.words])
    return spans, spans_to_docile_predictions(doc, spans, confidences)


def run_prediction_pipeline(
    model_path: Path,
    dataset_name: str,
    model_type: str = "auto",
    split: str = "test",
    limit: int | None = None,
    output_json: Path | None = None,
    device_str: str = "auto",
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute prediction pipeline across RoBERTa or GNN models."""
    if torch is None:
        raise RuntimeError("PyTorch is required for prediction.")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive number of documents")

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu"
    )

    adapter_cls = ADAPTERS[dataset_name]
    adapter = load_adapter(dataset_name, adapter_cls)

    source_split = split
    aliases = {"val": "validation", "validation": "val"}
    if source_split not in adapter.splits() and aliases.get(split) in adapter.splits():
        source_split = aliases[split]
    if source_split not in adapter.splits():
        raise ValueError(f"Split '{split}' not found in dataset '{dataset_name}'. Available: {adapter.splits()}")

    has_gt = getattr(adapter, "has_ground_truth", lambda s: True)(source_split)

    cache_dir = cache_dir or Path(__file__).resolve().parents[2] / ".cache"
    vocab_file = cache_dir / f"{dataset_name}-vocabulary.json"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "vocabulary" in ckpt:
        vocab = ckpt["vocabulary"]
    elif vocab_file.exists():
        vocab, _ = load_vocabulary(vocab_file)
    else:
        raise FileNotFoundError(f"Checkpoint has no vocabulary and cache is missing: {vocab_file}")
    inv_vocab = {idx: name for name, idx in vocab.items()}

    # Detect model type
    detected_model = model_type
    if detected_model == "auto":
        cfg = ckpt.get("config", {})
        if ckpt.get("model_type") == "layoutlmv3":
            detected_model = "layoutlmv3"
        elif isinstance(cfg, InvoiceGNNConfig) or (isinstance(cfg, dict) and "geo_dim" in cfg) or "gnn" in str(model_path).lower():
            detected_model = "gnn"
        else:
            detected_model = "roberta"

    tokenizer = None
    if detected_model == "layoutlmv3":
        from app.models_dl.layoutlmv3 import load_layoutlmv3_checkpoint
        del ckpt
        model, tokenizer, vocab = load_layoutlmv3_checkpoint(model_path, device)
        inv_vocab = {idx: name for name, idx in vocab.items()}
    elif detected_model == "gnn":
        validate_gnn_checkpoint(ckpt)
        raw_cfg = ckpt["config"]
        gnn_config = raw_cfg if isinstance(raw_cfg, InvoiceGNNConfig) else InvoiceGNNConfig(**raw_cfg)
        model = InvoiceGNN(gnn_config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
    else:
        config = RoBERTaWithLayoutConfig.from_dict(ckpt["config"])
        backbone = (
            RobertaModel(RobertaConfig.from_dict(ckpt["backbone_config"]))
            if "backbone_config" in ckpt else None
        )
        model = RoBERTaWithLayoutForTokenClassification(config=config, roberta_model=backbone)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        tokenizer_source = model_path.parent / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_source) if tokenizer_source.is_dir() else config.model_name_or_path or "roberta-base",
            add_prefix_space=True,
        )

    doc_ids = adapter.document_ids(source_split)
    expected_doc_ids = list(doc_ids)
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Dataset split contains duplicate document IDs")
    if not doc_ids:
        raise ValueError(f"Dataset split '{split}' is empty")
    if limit is not None:
        doc_ids = doc_ids[:limit]

    profiler = InferenceProfiler(device=device)
    profiler.start()

    all_docile_predictions: dict[str, list[dict[str, Any]]] = {}
    pred_entities_batch: list[list[EntitySpan]] = []
    true_entities_batch: list[list[EntitySpan]] = []
    target_values: dict[str, list[dict[str, Any]]] = {}

    for doc_id in doc_ids:
        doc = adapter.load(doc_id)
        if detected_model == "layoutlmv3":
            pred_spans, docile_preds = predict_document_layoutlmv3(
                model, doc, tokenizer, vocab, inv_vocab, device, adapter.get_image)
        elif detected_model == "gnn":
            pred_spans, docile_preds = predict_document_gnn(
                model=model,
                doc=doc,
                vocab=vocab,
                inv_vocab=inv_vocab,
                device=device,
            )
        else:
            pred_spans, docile_preds = predict_document_roberta(
                model=model,
                doc=doc,
                tokenizer=tokenizer,
                vocab=vocab,
                inv_vocab=inv_vocab,
                device=device,
            )
        all_docile_predictions[doc_id] = docile_preds

        if has_gt:
            target_values[doc_id] = [
                {"fieldtype": field.fieldtype, "text": field.text} for field in doc.fields
            ]
            pred_entities_batch.append(pred_spans)
            # Preserve unseen field types instead of silently mapping them to O.
            word_gt_tags, _ = label_document(doc)
            gt_spans = extract_entities_from_bio(
                word_gt_tags, texts=[word.text for word in doc.words],
                pages=[word.page for word in doc.words],
            )
            true_entities_batch.append(gt_spans)

    perf = profiler.stop(num_documents=len(doc_ids))

    results: dict[str, Any] = {
        "model": detected_model.upper(),
        "dataset": dataset_name,
        "split": split,
        "source_split": source_split,
        "num_documents": len(doc_ids),
        "expected_num_documents": len(expected_doc_ids),
        "complete": len(doc_ids) == len(expected_doc_ids),
        "protocol": "ocr-bio-exact-span-v2",
        "split_sha256": hashlib.sha256(json.dumps(expected_doc_ids).encode()).hexdigest(),
        "checkpoint_sha256": file_sha256(model_path),
        "has_ground_truth": has_gt,
        "latency": perf["latency"],
        "throughput": perf.get("throughput", 0.0),
        "perf": perf,
    }
    if detected_model == "roberta":
        results["window_policy"] = WINDOW_POLICY
    elif detected_model == "layoutlmv3":
        from app.data.layoutlmv3_dataset import WINDOW_POLICY as layout_policy
        results["window_policy"] = layout_policy
        results["image_input"] = "original-page-rgb"
        results["ocr_input"] = "dataset-provided"

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(all_docile_predictions, fh, indent=2)
        results["prediction_sha256"] = file_sha256(output_json)

    if has_gt and true_entities_batch:
        entity_metrics = compute_entity_metrics(
            pred_entities_batch=pred_entities_batch,
            true_entities_batch=true_entities_batch,
            match_mode="exact",
        )
        results["precision"] = float(entity_metrics["precision"])
        results["recall"] = float(entity_metrics["recall"])
        results["f1"] = float(entity_metrics["f1"])
        results["entity_metrics"] = entity_metrics
        results["value_metrics"] = compute_value_metrics(all_docile_predictions, target_values)

    if output_json:
        output_json.with_suffix(".metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    return results


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prediction_filename(model: str, dataset: str, split: str, limit: int | None) -> str:
    """Keep quick checks from overwriting a full benchmark export."""
    suffix = f"_sample{limit}" if limit is not None else ""
    return f"predictions_{model}_{dataset}_{split}{suffix}.json"


def print_prediction_summary(results: dict[str, Any], output_path: Path | None = None) -> None:
    """Format and print inference results to console for CLI callers."""
    model = results.get("model", "MODEL")
    dataset = str(results.get("dataset", "")).upper()
    split = str(results.get("split", "")).upper()
    num_docs = results.get("num_documents", 0)

    print(f"\n=== Inference Summary: {model} on {dataset} [{split}] ({num_docs} docs) ===")
    if output_path:
        print(f"Predictions saved to: {output_path}")

    if results.get("has_ground_truth") and "f1" in results:
        p = results.get("precision", 0.0) * 100
        r = results.get("recall", 0.0) * 100
        f1 = results.get("f1", 0.0) * 100
        print(f"Precision:  {p:.2f}%")
        print(f"Recall:     {r:.2f}%")
        print(f"F1:         {f1:.2f}%")

    perf = results.get("perf", {})
    lat = perf.get("latency", results.get("latency", 0.0))
    tp = perf.get("throughput", results.get("throughput", 0.0))
    print(f"Latency:    {lat:.1f} ms")
    print(f"Throughput: {tp:.1f} docs/sec")


def _resolve_checkpoint(model_name: str, dataset_name: str, checkpoints_dir: Path | None = None) -> Path | None:
    ckpt_dir = checkpoints_dir or (Path(__file__).resolve().parents[2] / "checkpoints")
    prefix = model_name if model_name != "auto" else "*"
    candidate_patterns = [
        f"{prefix}_{dataset_name}_best.pt",
        f"{dataset_name}_{prefix}_best.pt",
    ]
    if model_name == "auto":
        candidate_patterns.append(f"*{dataset_name}*best*.pt")
    for pat in candidate_patterns:
        matches = sorted(ckpt_dir.glob(pat))
        if matches:
            return matches[0]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="auto", choices=["auto", "roberta", "gnn", "layoutlmv3"], help="Model architecture")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to trained .pt checkpoint file")
    parser.add_argument("--dataset", default="all", choices=["all", "docile", "cord", "sroie"], help="Dataset name ('all', 'sroie', 'cord', 'docile')")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (test, val, train, test_official)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path to save benchmark predictions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to evaluate")
    parser.add_argument("--device", default="auto", help="Device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory for vocabulary")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory to save prediction reports")
    parser.add_argument("--all", action="store_true", help="Evaluate all models across all datasets")
    args = parser.parse_args(argv)

    reports_root = args.report_dir or (Path(__file__).resolve().parents[2] / "reports")
    pred_dir = reports_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.dataset == "all":
        models_to_run = ["gnn", "roberta", "layoutlmv3"] if args.model == "auto" else [args.model]
        datasets_to_run = ["sroie", "cord", "docile"] if (args.all or args.dataset == "all") else [args.dataset]
        all_results = []

        for m in models_to_run:
            for d in datasets_to_run:
                ckpt = _resolve_checkpoint(m, d)
                if not ckpt:
                    print(f"[SKIP] No checkpoint found for {m.upper()} on {d.upper()}")
                    continue
                out_json = pred_dir / prediction_filename(m, d, args.split, args.limit)
                print(f"\n>>> Running {m.upper()} on {d.upper()} [{args.split.upper()}] with {ckpt.name}...")
                try:
                    res = run_prediction_pipeline(
                        model_path=ckpt,
                        dataset_name=d,
                        model_type=m,
                        split=args.split,
                        limit=args.limit,
                        output_json=out_json,
                        device_str=args.device,
                        cache_dir=args.cache_dir,
                    )
                    all_results.append(res)
                except Exception as exc:
                    print(f"[ERROR] Failed {m.upper()} on {d.upper()}: {exc}")

        if all_results:
            print("\n" + "=" * 80)
            print(f"{'Model':<10} | {'Dataset':<10} | {'Split':<12} | {'Precision':<11} | {'Recall':<10} | {'F1':<10} | {'Latency':<10}")
            print("=" * 80)
            for r in all_results:
                p_str = f"{r.get('precision', 0.0)*100:.2f}%" if "precision" in r else "—"
                r_str = f"{r.get('recall', 0.0)*100:.2f}%" if "recall" in r else "—"
                f_str = f"{r.get('f1', 0.0)*100:.2f}%" if "f1" in r else "—"
                lat_str = f"{r.get('latency', 0.0):.1f} ms"
                print(f"{r['model']:<10} | {r['dataset'].upper():<10} | {r['split']:<12} | {p_str:<11} | {r_str:<10} | {f_str:<10} | {lat_str:<10}")
            print("=" * 80)
        return 0

    checkpoint_path = args.checkpoint
    detected_model = args.model

    if checkpoint_path is None:
        checkpoint_path = _resolve_checkpoint(args.model, args.dataset)
        if checkpoint_path is None:
            prefix = args.model if args.model != "auto" else "*"
            ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
            print(
                f"[ERROR] No checkpoint found matching '{prefix}_{args.dataset}_best.pt' in {ckpt_dir}.\n"
                f"Please provide --checkpoint explicitly or train the model first.",
                file=sys.stderr,
            )
            return 1
        print(f"[Auto-Checkpoint] Found: {checkpoint_path.name}")

    if detected_model == "auto" and checkpoint_path is not None:
        if "gnn" in checkpoint_path.name.lower():
            detected_model = "gnn"
        elif "roberta" in checkpoint_path.name.lower():
            detected_model = "roberta"

    model_tag = detected_model.lower() if detected_model != "auto" else "model"
    default_output = args.output or (pred_dir / prediction_filename(model_tag, args.dataset, args.split, args.limit))

    results = run_prediction_pipeline(
        model_path=checkpoint_path,
        dataset_name=args.dataset,
        model_type=detected_model,
        split=args.split,
        limit=args.limit,
        output_json=default_output,
        device_str=args.device,
        cache_dir=args.cache_dir,
    )
    print_prediction_summary(results, default_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
