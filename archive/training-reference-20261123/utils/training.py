import argparse
import csv
import json
import math

import torch
from transformers import get_linear_schedule_with_warmup

from utils.common import DATASETS, close_logger, setup_logger
from utils.plots import save_train_plots


def weights(model):
    if hasattr(model, "peft_config"):
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(model)
    return model.state_dict()


def load_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if hasattr(model, "peft_config"):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state)
    else:
        model.load_state_dict(state)


def fit(model, loader, evaluate, checkpoint_dir, out_dir, *, epochs=30,
        accum_steps=1, lr=3e-5, min_epochs=5, patience=5, resume=False):
    if min(epochs, accum_steps, min_epochs, patience) < 1 or not len(loader):
        raise ValueError("Training requires positive settings and nonempty data")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "last_checkpoint.pt"
    if checkpoint_file.exists() and not resume:
        raise FileExistsError(f"Existing run: {checkpoint_dir}; resume or move it first")
    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim > 1], "weight_decay": 0.01},
        {"params": [p for p in parameters if p.ndim <= 1], "weight_decay": 0.0},
    ], lr=lr)
    total_steps = math.ceil(len(loader) / accum_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    config = dict(epochs=epochs, accum_steps=accum_steps, lr=lr, min_epochs=min_epochs,
                  patience=patience, batches=len(loader), batch_size=loader.batch_size)
    history, best_f1, best_epoch, stale, start, updates, skips = [], -1.0, 0, 0, 1, 0, 0
    if resume:
        saved = torch.load(checkpoint_file, map_location=device, weights_only=True)
        if saved["config"] != config:
            raise ValueError("Resume settings differ from the saved run")
        load_weights(model, checkpoint_dir / "last_weights.pt", device)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, best_f1, best_epoch, stale = (saved[k] for k in ("history", "best_f1", "best_epoch", "stale"))
        start, updates, skips = saved["epoch"] + 1, saved["updates"], saved["skips"]
        torch.set_rng_state(saved["rng"].cpu())
        if amp and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
    logger = setup_logger(out_dir)
    try:
        logger.info(f"Config: {config}; device={device}; input=original annotations")
        for epoch in range(start, epochs + 1):
            if stale >= patience:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for step, batch in enumerate(loader, 1):
                batch = {key: value.to(device) for key, value in batch.items()}
                group_size = min(accum_steps, len(loader) - ((step - 1) // accum_steps) * accum_steps)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: epoch={epoch}, batch={step}")
                total_loss += loss.item()
                scaler.scale(loss / group_size).backward()
                if step % accum_steps == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    after = scaler.get_scale()
                    optimizer.zero_grad(set_to_none=True)
                    if after < before:
                        skips += 1
                        logger.warning(f"AMP skip: epoch={epoch} batch={step} scale={before}->{after}")
                    else:
                        updates += 1
                        scheduler.step()
            val_loss, metrics = evaluate()
            if not math.isfinite(val_loss) or any(not math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Non-finite validation result")
            row = dict(epoch=epoch, train_loss=total_loss / len(loader), val_loss=val_loss,
                       **{key: metrics[key] for key in ("precision", "recall", "f1")})
            history.append(row)
            logger.info(str(row))
            if metrics["f1"] > best_f1:
                best_f1, best_epoch, stale = metrics["f1"], epoch, 0
                torch.save(weights(model), checkpoint_dir / "best_weights.pt")
            elif epoch >= min_epochs:
                stale += 1
            torch.save(weights(model), checkpoint_dir / "last_weights.pt")
            torch.save(dict(epoch=epoch, config=config, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            history=history, best_f1=best_f1, best_epoch=best_epoch, stale=stale,
                            updates=updates, skips=skips, rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if amp else None), checkpoint_file)
            with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
        if history:
            save_train_plots(history, out_dir)
        summary = dict(best_epoch=best_epoch, val_f1=best_f1, epochs=len(history),
                       optimizer_updates=updates, amp_skips=skips, config=config)
        (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        load_weights(model, checkpoint_dir / "best_weights.pt", device)
        return summary
    finally:
        close_logger(logger)


def run_cli(train_model, batch_size, accum_steps, lr):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--accum_steps", type=int, default=accum_steps)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = vars(parser.parse_args())
    args["dataset_name"] = args.pop("dataset")
    train_model(**args)
