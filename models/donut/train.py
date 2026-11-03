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
    DonutProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)

from models.donut.dataset import DonutDataCollator, DonutDataset
from utils.common import (
    DATASETS,
    close_logger,
    get_repo_dir,
    get_cache_dir,
    get_data_dir,
    set_seed,
    setup_logger,
)
from utils.generation import parse_output
from utils.baseline_protocol import verify_baseline_data
from utils.metrics import evaluate_dataset
from utils.plots import save_train_plots


def build_model(weights_dir, device):
    processor = DonutProcessor.from_pretrained(weights_dir)
    processor.image_processor.size = {"height": 1280, "width": 960}
    model = VisionEncoderDecoderModel.from_pretrained(weights_dir)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None
    model.decoder.gradient_checkpointing_enable()
    model.config.decoder.use_cache = False
    return model.to(device), processor


def train_epoch(
    model: VisionEncoderDecoderModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int = 30,
    accum_steps: int = 8,
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
        pixel_values = batch["pixel_values"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            raw_loss = outputs.loss
            group_size = min(accum_steps, len(loader) - ((step - 1) // accum_steps) * accum_steps)
            loss = raw_loss / group_size

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
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
            else:
                logging.getLogger("train").warning(
                    f"AMP skip: epoch={epoch} batch={step} scale={scale_before}->{scale_after}")

        total_loss += raw_loss.item()
        progress_loader.set_postfix(loss=f"{raw_loss.item():.4f}")

    return total_loss / len(loader)


def evaluate_epoch(
    model: VisionEncoderDecoderModel,
    loader: DataLoader,
    inference_loader: DataLoader,
    samples: list[dict],
    processor: DonutProcessor,
    device: torch.device,
    desc: str,
    max_new_tokens: int = 1535,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    preds: list[dict] = []
    use_amp = device.type == "cuda"
    tokenizer = processor.tokenizer

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{desc} Loss", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
            total_loss += outputs.loss.item()

        # Generate from the image and BOS only, without teacher-forced target tokens.
        for batch in tqdm(inference_loader, desc=f"{desc} Generate", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            prefix_width = batch["decoder_input_ids"].shape[1]
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                sequences = model.generate(
                    **batch, max_new_tokens=max_new_tokens,
                    do_sample=False, num_beams=1, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(sequences[:, prefix_width:], skip_special_tokens=True)
            preds.extend(parse_output(text) for text in texts)

    avg_loss = total_loss / len(loader)
    gts = [sample["gt"] for sample in samples]
    metrics = evaluate_dataset(gts, preds)
    return avg_loss, metrics


def evaluate_test(
    model: VisionEncoderDecoderModel,
    loader: DataLoader,
    inference_loader: DataLoader,
    samples: list[dict],
    processor: DonutProcessor,
    device: torch.device,
    dataset_name: str,
    checkpoint_dir: Path,
    out_dir: Path,
    logger: logging.Logger,
    max_new_tokens: int = 1535,
) -> None:
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_weights.pt", map_location=device, weights_only=True)
    )
    loss, metrics = evaluate_epoch(
        model, loader, inference_loader, samples, processor, device, "[Test]", max_new_tokens
    )

    results = {
        "model": "donut",
        "dataset": dataset_name,
        "data_protocol": "completed-baselines-20261024",
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
    batch_size: int = 1,
    accum_steps: int = 8,
    lr: float = 3e-5,
    min_epochs: int = 5,
    patience: int = 5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    resume: bool = False,
    test_only: bool = False,
    max_new_tokens: int = 1535,
) -> None:
    set_seed()
    repo_dir = get_repo_dir()
    checkpoint_dir = repo_dir / "checkpoints" / dataset_name / "donut"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    out_dir = repo_dir / "results" / dataset_name / "donut"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    try:
        logger.info(
            f"Config: dataset={dataset_name}, epochs={epochs}, batch_size={batch_size}, "
            f"accum_steps={accum_steps}, lr={lr:g}, min_epochs={min_epochs}, patience={patience}, "
            f"max_new_tokens={max_new_tokens}"
        )

        cache_dir = get_cache_dir(dataset_name)
        verify_baseline_data(dataset_name, cache_dir)
        logger.info("Data protocol: completed-baselines-20261024; prepared splits verified")
        data_dir = get_data_dir(dataset_name)
        weights_dir = repo_dir / "weights" / "donut"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = build_model(weights_dir, device)
        if not 1 <= max_new_tokens < model.config.decoder.max_position_embeddings:
            raise ValueError("max_new_tokens must fit the decoder positions after the BOS token")
        collator = DonutDataCollator(processor=processor)
        inference_collator = DonutDataCollator(processor=processor, training=False)

        if test_only:
            test_ds = DonutDataset(cache_dir / "test.jsonl", processor, data_dir,
                                   max_new_tokens=max_new_tokens)
            test_inference_ds = DonutDataset(cache_dir / "test.jsonl", processor, data_dir,
                                             training=False, max_new_tokens=max_new_tokens)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
            test_inference_loader = DataLoader(test_inference_ds, batch_size=batch_size, shuffle=False,
                                               collate_fn=inference_collator)
            evaluate_test(
                model, test_loader, test_inference_loader, test_ds.samples, processor,
                device, dataset_name, checkpoint_dir, out_dir, logger, max_new_tokens,
            )
            return

        train_ds = DonutDataset(cache_dir / "train.jsonl", processor, data_dir,
                                max_new_tokens=max_new_tokens)
        val_ds = DonutDataset(cache_dir / "val.jsonl", processor, data_dir,
                              max_new_tokens=max_new_tokens)
        val_inference_ds = DonutDataset(cache_dir / "val.jsonl", processor, data_dir,
                                        training=False, max_new_tokens=max_new_tokens)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
        val_inference_loader = DataLoader(val_inference_ds, batch_size=batch_size, shuffle=False,
                                          collate_fn=inference_collator)

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
        start_epoch = 1
        best_epoch = 0
        best_f1 = -1.0
        patience_counter = 0

        checkpoint_file = checkpoint_dir / "last_checkpoint.pt"
        if checkpoint_file.exists() and not resume:
            raise FileExistsError(f"Existing run: {checkpoint_dir}; use resume or move the previous run")
        if resume and not checkpoint_file.exists():
            raise FileNotFoundError(checkpoint_file)
        if resume and checkpoint_file.exists():
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
            if checkpoint.get("data_protocol") != "completed-baselines-20261024":
                raise ValueError("Resume checkpoint uses a different data protocol; start a separate run")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_f1 = checkpoint["best_f1"]
            best_epoch = checkpoint["best_epoch"]
            patience_counter = checkpoint["patience_counter"]
            history = checkpoint["history"]
            if start_epoch > epochs or patience_counter >= patience:
                logger.info(
                    f"Resume -> Training already completed at epoch {checkpoint['epoch']:02d}/{epochs:02d} (Best F1: {best_f1:.4f})."
                )
                return

            logger.info(
                f"Resume -> Epoch {checkpoint['epoch']:02d}/{epochs:02d} (Best F1: {best_f1:.4f})"
            )

        for epoch in range(start_epoch, epochs + 1):
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
                val_inference_loader,
                val_ds.samples,
                processor,
                device,
                f"Epoch {epoch:02d}/{epochs:02d} [Val]",
                max_new_tokens,
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
            elif epoch >= min_epochs:
                patience_counter += 1

            torch.save(
                {
                    "epoch": epoch,
                    "data_protocol": "completed-baselines-20261024",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_f1": best_f1,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter,
                    "history": history,
                },
                checkpoint_file,
            )

            if epoch >= min_epochs and patience_counter >= patience:
                logger.info(
                    f"Early Stop -> Epoch {epoch:02d}/{epochs:02d} (Best F1: {best_f1:.4f} at epoch {best_epoch:02d})"
                )
                break

        save_history(history, out_dir)

    finally:
        close_logger(logger)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=1535)
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
            min_epochs=args.min_epochs,
            patience=args.patience,
            resume=args.resume,
            test_only=args.test_only,
            max_new_tokens=args.max_new_tokens,
        )
