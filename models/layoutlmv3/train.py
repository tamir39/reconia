import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from models.layoutlmv3.dataset import LayoutLMv3Dataset, LayoutLMv3Collator, decode_predictions
from utils.common import get_cache_dir, get_data_dir, get_repo_dir, load_labels, set_seed
from utils.metrics import evaluate_dataset
from utils.training import fit, load_weights, run_cli


@torch.no_grad()
def evaluate(model, loader, tokenizer, id2label, device):
    model.eval()
    loss, predictions = 0.0, []
    for batch in loader:
        output = model(**{key: value.to(device) for key, value in batch.items()})
        loss += output.loss.item()
        predictions.extend(output.logits.argmax(-1).cpu().tolist())
    samples = loader.dataset.samples
    fields = decode_predictions(predictions, samples, tokenizer, id2label)
    return loss / len(loader), evaluate_dataset([sample["gt"] for sample in samples], fields)


def train_model(dataset_name, epochs=30, batch_size=4, accum_steps=2, lr=3e-5,
                min_epochs=5, patience=5, resume=False, test_only=False):
    set_seed()
    root = get_repo_dir()
    cache, data = get_cache_dir(dataset_name), get_data_dir(dataset_name)
    checkpoint = root / "checkpoints" / dataset_name / "layoutlmv3"
    out = root / "results" / dataset_name / "layoutlmv3"
    labels = load_labels(dataset_name)
    label2id, id2label = {label: index for index, label in enumerate(labels)}, dict(enumerate(labels))
    processor = AutoProcessor.from_pretrained(root / "weights" / "layoutlmv3", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        root / "weights" / "layoutlmv3", num_labels=len(labels), id2label=id2label, label2id=label2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collator = LayoutLMv3Collator(processor.tokenizer)

    def loader(split, shuffle=False):
        dataset = LayoutLMv3Dataset(cache / f"{split}.jsonl", processor, label2id, data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

    if test_only:
        load_weights(model, checkpoint / "best_weights.pt", device)
        loss, metrics = evaluate(model, loader("test"), processor.tokenizer, id2label, device)
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_metrics.json").write_text(json.dumps(dict(model="layoutlmv3", dataset=dataset_name,
                                                            loss=loss, **metrics), indent=2), encoding="utf-8")
        return metrics
    val = loader("val")
    return fit(model, loader("train", True),
               lambda: evaluate(model, val, processor.tokenizer, id2label, device), checkpoint, out,
               epochs=epochs, accum_steps=accum_steps, lr=lr, min_epochs=min_epochs,
               patience=patience, resume=resume)


if __name__ == "__main__":
    run_cli(train_model, 4, 2, 3e-5)
