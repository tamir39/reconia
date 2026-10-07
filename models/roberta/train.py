import argparse
import csv
import gc
import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    RobertaForTokenClassification,
    get_linear_schedule_with_warmup,
)

from tools.common import get_repo_dir, get_labels, set_seed, setup_logger, close_logger
from tools.plots import save_train_plots
from evaluate.metrics import evaluate_dataset
from models.roberta.dataset import (
    RobertaDataset,
    reconstruct_predictions,
)


def train_epoch(
    model: RobertaForTokenClassification,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    scheduler: LRScheduler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int,
    accum_steps: int = 2,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    progress_loader = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
        leave=False,
    )
    for step, batch in enumerate(progress_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )
            loss = outputs.loss / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad()
            if scale_before <= scale_after:
                scheduler.step()

        total_loss += loss.item() * accum_steps
        progress_loader.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return total_loss / len(loader) if len(loader) > 0 else 0.0


def evaluate_epoch(
    model: RobertaForTokenClassification,
    loader: DataLoader,
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict[int, str],
    device: torch.device,
    desc: str = "Val",
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds: list[list[int]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )
            total_loss += outputs.loss.item()

            logits = outputs.logits
            batch_pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(batch_pred_ids)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    reconstructed = reconstruct_predictions(all_preds, samples, tokenizer, id2label)
    gts = [s.get("gt", {}) for s in samples]
    metrics = evaluate_dataset(reconstructed, gts)

    return avg_loss, metrics


def save_history(history: list[dict], out_dir: Path) -> None:
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(history)
    save_train_plots(history, out_dir)


def evaluate_test(
    model: RobertaForTokenClassification,
    loader: DataLoader,
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict[int, str],
    device: torch.device,
    dataset: str,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    test_loss, test_metrics = evaluate_epoch(model, loader, samples, tokenizer, id2label, device, desc="Test")

    final_results = {
        "model": "roberta",
        "dataset": dataset,
        "test_loss": round(test_loss, 4),
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2)

    logger.info(
        "Test -> Loss: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        test_loss,
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )


def train_model(
    dataset: str = "sroie",
    epochs: int = 30,
    batch_size: int = 8,
    accum_steps: int = 1,
    lr: float = 3e-5,
    patience: int = 5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> None:
    set_seed()
    repo_dir = get_repo_dir()
    out_dir = repo_dir / "results" / dataset / "roberta"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = repo_dir / "weights" / "roberta"
    model_source = str(weights_path) if weights_path.exists() else "roberta-base"

    logger = setup_logger(out_dir)
    try:
        cache_dir = repo_dir / ".cache" / dataset
        labels = get_labels(cache_dir)
        label2id = {lbl: idx for idx, lbl in enumerate(labels)}
        id2label = {idx: lbl for idx, lbl in enumerate(labels)}

        tokenizer = AutoTokenizer.from_pretrained(model_source, add_prefix_space=True)
        collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        train_ds = RobertaDataset(cache_dir / "train.jsonl", tokenizer, label2id)
        val_ds = RobertaDataset(cache_dir / "val.jsonl", tokenizer, label2id)
        test_ds = RobertaDataset(cache_dir / "test.jsonl", tokenizer, label2id)

        logger.info(
            "Config: dataset=%s, epochs=%d, batch_size=%d, accum_steps=%d, lr=%s, patience=%d",
            dataset,
            epochs,
            batch_size,
            accum_steps,
            f"{lr:g}",
            patience,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = RobertaForTokenClassification.from_pretrained(
            model_source,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        model.to(device)

        no_decay = {"bias", "layernorm"}
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(param_groups, lr=lr)
        total_steps = (len(train_loader) + accum_steps - 1) // accum_steps * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_f1 = -1.0
        best_epoch = 1
        patience_counter = 0
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler, scheduler, device, use_amp, epoch, epochs, accum_steps
            )
            val_loss, val_metrics = evaluate_epoch(
                model,
                val_loader,
                val_ds.samples,
                tokenizer,
                id2label,
                device,
                desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]",
            )

            is_best = val_metrics["f1"] > best_f1
            logger.info(
                "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
            )

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
            })

            torch.save(model.state_dict(), out_dir / "last.pt")

            if is_best:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), out_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopped at epoch %d (Best F1: %.4f at epoch %d)", epoch, best_f1, best_epoch)
                    break

        save_history(history, out_dir)
        evaluate_test(
            model,
            test_loader,
            test_ds.samples,
            tokenizer,
            id2label,
            device,
            dataset,
            out_dir,
            logger,
        )
    finally:
        close_logger(logger)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


DATASETS = ("sroie", "cord_v2", "wildreceipt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for target_dataset in target_datasets:
        train_model(
            dataset=target_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            lr=args.lr,
            patience=args.patience,
        )