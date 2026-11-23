import json

import torch
from torch.utils.data import DataLoader

from utils.common import get_cache_dir, get_data_dir, get_repo_dir, load_labels, set_seed
from utils.metrics import evaluate_dataset
from utils.training import fit, load_weights


def parse_output(text, *, strict=False):
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(values, list)
            or any(not isinstance(item, str) for item in values) for key, values in value.items()
        ):
            raise ValueError("Expected a JSON object mapping fields to lists of strings")
        return value
    except (ValueError, TypeError) as error:
        if strict:
            raise ValueError(str(error)) from error
        return {}


@torch.no_grad()
def evaluate(model, loss_loader, inference_loader, processor, device, name, max_new_tokens):
    model.eval()
    total_loss, predictions = 0.0, []
    amp = device.type == "cuda"
    for batch in loss_loader:
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            output = model(**{key: value.to(device) for key, value in batch.items()})
        total_loss += output.loss.item()
    for batch in inference_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prefix_width = batch["input_ids"].shape[1] if name == "qwen_vl" else batch["decoder_input_ids"].shape[1]
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            sequences = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                       num_beams=1, use_cache=True,
                                       pad_token_id=processor.tokenizer.pad_token_id,
                                       eos_token_id=processor.tokenizer.eos_token_id)
        texts = processor.tokenizer.batch_decode(sequences[:, prefix_width:], skip_special_tokens=True)
        predictions.extend(parse_output(text) for text in texts)
    gts = [sample["gt"] for sample in loss_loader.dataset.samples]
    return total_loss / len(loss_loader), evaluate_dataset(gts, predictions)


def train_generation(name, dataset_class, collator_class, build_model, dataset_name, *,
                     epochs, batch_size, accum_steps, lr, min_epochs, patience,
                     resume, test_only, max_new_tokens=2048):
    set_seed()
    root = get_repo_dir()
    checkpoint = root / "checkpoints" / dataset_name / name
    out = root / "results" / dataset_name / name
    fields = sorted({label[2:] for label in load_labels(dataset_name) if label != "O"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build_model(root / "weights" / name, device)

    def loader(split, training=True, shuffle=False):
        dataset = dataset_class(get_cache_dir(dataset_name) / f"{split}.jsonl",
                                processor, get_data_dir(dataset_name), fields=fields,
                                training=training, max_new_tokens=max_new_tokens)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collator_class(processor, training=training))

    if test_only:
        load_weights(model, checkpoint / "best_weights.pt", device)
        loss, metrics = evaluate(model, loader("test"), loader("test", False), processor,
                                 device, name, max_new_tokens)
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_metrics.json").write_text(json.dumps(dict(model=name, dataset=dataset_name,
                                                            loss=loss, **metrics), indent=2), encoding="utf-8")
        return metrics
    val_loss, val_inference = loader("val"), loader("val", False)
    return fit(model, loader("train", shuffle=True),
               lambda: evaluate(model, val_loss, val_inference, processor, device, name, max_new_tokens),
               checkpoint, out, epochs=epochs, accum_steps=accum_steps, lr=lr,
               min_epochs=min_epochs, patience=patience, resume=resume)
