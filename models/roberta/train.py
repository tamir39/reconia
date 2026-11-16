import argparse
import csv
import gc
import json
import logging
import os
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

from models.roberta.dataset import RobertaDataset, decode_predictions
from utils.common import (
    DATASETS,
    close_logger,
    get_repo_dir,
    load_labels,
    set_seed,
    setup_logger,
)
from utils.metrics import evaluate_dataset
from utils.plots import save_train_plots


def train_epoch(
    model: RobertaForTokenClassification,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int = 30,
    accum_steps: int = 1,
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
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(**batch)
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

    return total_loss / len(loader)


def evaluate_epoch(
    model: RobertaForTokenClassification,
    loader: DataLoader,
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict[int, str],
    device: torch.device,
    use_amp: bool,
    desc: str,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_pred_ids: list[list[int]] = []

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            all_pred_ids.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())

    avg_loss = total_loss / len(loader)
    preds = decode_predictions(all_pred_ids, samples, tokenizer, id2label)
    gts = [sample["gt"] for sample in samples]
    metrics = evaluate_dataset(gts, preds)

    return avg_loss, metrics


def evaluate_test(
    model: RobertaForTokenClassification,
    loader: DataLoader,
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict[int, str],
    device: torch.device,
    use_amp: bool,
    dataset_name: str,
    checkpoint_dir: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_weights.pt", map_location="cpu", weights_only=True)
    )
    loss, metrics = evaluate_epoch(
        model, loader, samples, tokenizer, id2label, device, use_amp, "[Test]"
    )

    results = {
        "model": "roberta",
        "dataset": dataset_name,
        "loss": loss,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    logger.info(
        f"Test -> Loss: {loss:.4f} | Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
    )


def save_history(history: list[dict], out_dir: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "precision", "recall", "f1"]
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    save_train_plots(history, out_dir)


def train_model(
    dataset_name: str,
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
    checkpoint_dir = repo_dir / "checkpoints" / dataset_name / "roberta"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    out_dir = repo_dir / "results" / dataset_name / "roberta"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    try:
        logger.info(
            f"\n--------------------------------------------------\n"
            f"Config -> {dataset_name}\n"
            f"  epochs       : {epochs}\n"
            f"  batch_size   : {batch_size} (accum_steps={accum_steps})\n"
            f"  lr           : {lr:g}\n"
            f"  patience     : {patience}\n"
            f"--------------------------------------------------"
        )

        cache_dir = repo_dir / ".cache" / dataset_name
        labels = load_labels(dataset_name)
        label2id = {lbl: idx for idx, lbl in enumerate(labels)}
        id2label = dict(enumerate(labels))

        weights_dir = repo_dir / "weights" / "roberta"
        tokenizer = AutoTokenizer.from_pretrained(weights_dir, add_prefix_space=True)
        collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)

        train_ds = RobertaDataset(cache_dir / "train.jsonl", tokenizer, label2id)
        val_ds = RobertaDataset(cache_dir / "val.jsonl", tokenizer, label2id)
        test_ds = RobertaDataset(cache_dir / "test.jsonl", tokenizer, label2id)

        eval_batch_size = batch_size * 2
        num_workers = min(2, os.cpu_count() or 0)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=False,
        )

        model = RobertaForTokenClassification.from_pretrained(
            weights_dir,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        no_decay = {"bias", "layernorm"}
        param_groups = [
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if not any(nd in name.lower() for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if any(nd in name.lower() for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(param_groups, lr=lr)

        total_steps = (len(train_loader) + accum_steps - 1) // accum_steps * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        history: list[dict] = []
        best_f1 = -1.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                use_amp,
                epoch,
                epochs,
                accum_steps,
            )
            val_loss, val_metrics = evaluate_epoch(
                model,
                val_loader,
                val_ds.samples,
                tokenizer,
                id2label,
                device,
                use_amp,
                f"Epoch {epoch:02d}/{epochs:02d} [Val]",
            )

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
            })

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "best_weights.pt")
            elif best_f1 > 0:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(
                    f"Early Stop -> Epoch {epoch:02d}/{epochs:02d} (Best F1: {best_f1:.4f} at epoch {best_epoch:02d})"
                )
                break

        if epoch == epochs:
            logger.info(
                f"Completed -> Epoch {epochs:02d}/{epochs:02d} (Best F1: {best_f1:.4f} at epoch {best_epoch:02d})"
            )

        save_history(history, out_dir)
        evaluate_test(
            model,
            test_loader,
            test_ds.samples,
            tokenizer,
            id2label,
            device,
            use_amp,
            dataset_name,
            checkpoint_dir,
            out_dir,
            logger,
        )
    finally:
        close_logger(logger)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset_name in datasets:
        train_model(
            dataset_name=dataset_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            lr=args.lr,
            patience=args.patience,
        )