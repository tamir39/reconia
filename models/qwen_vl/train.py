import argparse
import csv
import gc
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from models.qwen_vl.dataset import (
    QwenVLDataCollator,
    QwenVLDataset,
    decode_predictions,
)
from utils.common import (
    DATASETS,
    close_logger,
    get_data_dir,
    get_repo_dir,
    load_labels,
    set_seed,
    setup_logger,
)
from utils.metrics import evaluate_dataset
from utils.plots import save_train_plots


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int = 25,
    accum_steps: int = 4,
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
        batch = {key: value.to(device) for key, value in batch.items() if key != "prompt_len"}

        with torch.amp.autocast(device_type=device.type):
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
    model: torch.nn.Module,
    loader: DataLoader,
    samples: list[dict],
    processor: AutoProcessor,
    device: torch.device,
    desc: str,
    max_new_tokens: int = 1024,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds: list[dict[str, list[str]]] = []
    pad_token_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else 0
    )

    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        for batch in tqdm(loader, desc=desc, leave=False):
            batch_gpu = {key: value.to(device) for key, value in batch.items()}
            batch_loss = {key: value for key, value in batch_gpu.items() if key != "prompt_len"}
            valid_lengths = (batch_gpu["labels"] != -100).sum(dim=1)
            batch_max_new_tokens = min(
                max_new_tokens,
                int(valid_lengths.max().item()) + 32,
            )

            outputs = model(**batch_loss)
            total_loss += outputs.loss.item()
            del outputs

            prompt_lengths = batch_gpu["prompt_len"].tolist()
            max_prompt_len = max(prompt_lengths)
            batch_size = len(prompt_lengths)

            prompt_input_ids = torch.full(
                (batch_size, max_prompt_len),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            prompt_attention_mask = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=torch.long,
                device=device,
            )

            for sample_index, prompt_len in enumerate(prompt_lengths):
                offset = max_prompt_len - prompt_len
                prompt_input_ids[sample_index, offset:] = (
                    batch_gpu["input_ids"][sample_index, :prompt_len]
                )
                prompt_attention_mask[sample_index, offset:] = (
                    batch_gpu["attention_mask"][sample_index, :prompt_len]
                )

            generated_ids = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=batch_gpu["pixel_values"],
                image_grid_thw=batch_gpu["image_grid_thw"],
                max_new_tokens=batch_max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            new_tokens = generated_ids[:, max_prompt_len:]
            all_preds.extend(decode_predictions(new_tokens, processor))

    avg_loss = total_loss / len(loader)
    gts = [sample.get("gt", {}) for sample in samples]
    metrics = evaluate_dataset(gts, all_preds)
    return avg_loss, metrics


def evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    samples: list[dict],
    processor: AutoProcessor,
    device: torch.device,
    dataset_name: str,
    checkpoint_dir: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    set_peft_model_state_dict(
        model,
        torch.load(checkpoint_dir / "best_weights.pt", map_location="cpu", weights_only=True),
    )
    loss, metrics = evaluate_epoch(
        model, loader, samples, processor, device, "[Test]"
    )

    results = {
        "model": "qwen_vl",
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


def get_adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def save_history(history: list[dict], out_dir: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "precision", "recall", "f1"]
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    save_train_plots(history, out_dir)


def train_model(
    dataset_name: str,
    epochs: int = 25,
    batch_size: int = 2,
    accum_steps: int = 4,
    lr: float = 1e-4,
    patience: int = 5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    resume: bool = False,
) -> None:
    set_seed()

    repo_dir = get_repo_dir()
    checkpoint_dir = repo_dir / "checkpoints" / dataset_name / "qwen_vl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    out_dir = repo_dir / "results" / dataset_name / "qwen_vl"
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
        data_dir = get_data_dir(dataset_name)
        labels = load_labels(dataset_name)

        weights_dir = repo_dir / "weights" / "qwen_vl"
        processor = AutoProcessor.from_pretrained(weights_dir)
        processor.image_processor.size = {
            "shortest_edge": 256 * 28 * 28,
            "longest_edge": 768 * 28 * 28,
        }
        collate_fn = QwenVLDataCollator(processor=processor)

        train_ds = QwenVLDataset(data_dir, cache_dir / "train.jsonl", processor, labels)
        val_ds = QwenVLDataset(data_dir, cache_dir / "val.jsonl", processor, labels)
        test_ds = QwenVLDataset(data_dir, cache_dir / "test.jsonl", processor, labels)

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

        device = torch.device("cuda")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "model.visual", "lm_head"],
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            weights_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=device,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.model.visual.to(torch.float32)
        model.model.visual.forward = torch.amp.custom_fwd(
            device_type=device.type, cast_inputs=torch.float32
        )(model.model.visual.forward)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=r".*layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        )
        model = get_peft_model(model, lora_config)

        no_decay = {"bias", "layernorm", "norm"}
        param_groups = [
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if not any(nd in name.lower() for nd in no_decay) and param.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if any(nd in name.lower() for nd in no_decay) and param.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(param_groups, lr=lr)

        total_steps = (len(train_loader) + accum_steps - 1) // accum_steps * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler()

        history: list[dict] = []
        start_epoch = 1
        best_f1 = -1.0
        best_epoch = 0
        patience_counter = 0

        checkpoint_file = checkpoint_dir / "last_checkpoint.pt"
        if resume and checkpoint_file.exists():
            ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            set_peft_model_state_dict(model, ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_f1 = ckpt["best_f1"]
            best_epoch = ckpt["best_epoch"]
            patience_counter = ckpt["patience_counter"]
            history = ckpt["history"]
            random.setstate(ckpt["python_rng_state"])
            np.random.set_state(ckpt["numpy_rng_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])

            if start_epoch > epochs or patience_counter >= patience:
                logger.info(f"Resume -> Completed at epoch {ckpt['epoch']:02d}/{epochs:02d} (Best F1: {best_f1:.4f} at epoch {best_epoch:02d})")
                return

            logger.info(f"Resume -> Epoch {ckpt['epoch']:02d}/{epochs:02d} (Best F1: {best_f1:.4f} at epoch {best_epoch:02d})")

        for epoch in range(start_epoch, epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                epochs,
                accum_steps,
            )
            val_loss, val_metrics = evaluate_epoch(
                model,
                val_loader,
                val_ds.samples,
                processor,
                device,
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
                torch.save(get_adapter_state(model), checkpoint_dir / "best_weights.pt")
            elif best_f1 > 0:
                patience_counter += 1

            if resume:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": get_adapter_state(model),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "best_f1": best_f1,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter,
                        "history": history,
                        "python_rng_state": random.getstate(),
                        "numpy_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state(),
                    },
                    checkpoint_file,
                )

            if patience_counter >= patience:
                logger.info(f"Early Stop -> Epoch {epoch:02d}/{epochs:02d} (Best F1: {best_f1:.4f} at epoch {best_epoch:02d})")
                break

            if epoch == epochs:
                logger.info(f"Completed -> Epoch {epochs:02d}/{epochs:02d} (Best F1: {best_f1:.4f} at epoch {best_epoch:02d})")

        save_history(history, out_dir)
        evaluate_test(
            model,
            test_loader,
            test_ds.samples,
            processor,
            device,
            dataset_name,
            checkpoint_dir,
            out_dir,
            logger,
        )
    finally:
        close_logger(logger)
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
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
            resume=args.resume,
        )