"""Checkpoint-backed inference using the code exported by each Kaggle run."""
import gc
import importlib
from functools import lru_cache

from demo.assets import (
    ROOT, assert_checkpoint, labels, model_assets,
)


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)


def read_image_text(image):
    from utils.common import normalize_box

    resized = image.copy()
    resized.thumbnail((2400, 2400))
    lines, _ = ocr_engine()(resized)
    return [
        {"text": text.strip(), "box": normalize_box(
            [coordinate for point in points for coordinate in point], resized.width, resized.height
        )}
        for points, text, _ in (lines or []) if text.strip()
    ]


@lru_cache(maxsize=15)
def baseline_module(model, name="dataset"):
    return importlib.import_module(f"models.{model}.{name}")


_LOADED = None


def load_model(dataset, name):
    global _LOADED
    import os
    import torch
    from transformers import (
        AutoConfig, AutoProcessor, AutoTokenizer, BrosForTokenClassification,
        LayoutLMv3ForTokenClassification, RobertaForTokenClassification, VisionEncoderDecoderModel,
    )

    if _LOADED is not None and _LOADED[:2] == (dataset, name):
        return _LOADED[2]
    _LOADED = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    path = assert_checkpoint(dataset, name)
    tags = labels(dataset, name)
    id2label = dict(enumerate(tags))
    assets = model_assets(name)
    requested = os.environ.get("RECONIA_DEVICE", "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("RECONIA_DEVICE must be cpu or cuda")
    device = torch.device(requested if requested == "cpu" or torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 6))
    if name == "pick":
        data = baseline_module(name)
        chars = data.build_char_vocab()
        _, id2label = data.build_tag_vocab(tags)
        architecture = baseline_module(name, "model")
        from torchvision.models import resnet50
        architecture.resnet50 = lambda **kwargs: resnet50(weights=None)
        model = architecture.PICKModel(vocab_size=len(chars), num_tags=len(id2label))
        processor = chars
    elif name == "donut":
        processor = AutoProcessor.from_pretrained(assets, local_files_only=True)
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<s_{dataset}>"]})
        config = AutoConfig.from_pretrained(assets, local_files_only=True)
        config.decoder.vocab_size = len(tokenizer)
        config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 0
        config.pad_token_id = tokenizer.pad_token_id
        model = VisionEncoderDecoderModel(config=config)
    else:
        config = AutoConfig.from_pretrained(assets, num_labels=len(tags), local_files_only=True)
        if name == "layoutlmv3":
            processor = AutoProcessor.from_pretrained(assets, apply_ocr=False, local_files_only=True)
            model = LayoutLMv3ForTokenClassification(config)
        else:
            options = {"add_prefix_space": True} if name == "roberta" else {}
            processor = AutoTokenizer.from_pretrained(assets, local_files_only=True, **options)
            cls = RobertaForTokenClassification if name == "roberta" else BrosForTokenClassification
            model = cls(config)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False)
    result = model.to(device).eval(), processor, id2label, device
    _LOADED = dataset, name, result
    return result


def donut_inputs(dataset, processor, image, device):
    pixels = processor(image, return_tensors="pt", do_resize=True,
                       size={"height": 1280, "width": 960})["pixel_values"].to(device)
    prefix = processor.tokenizer(f"<s_{dataset}>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    return pixels, prefix


def donut_generate(dataset, model, processor, image, device):
    import torch

    pixels, prefix = donut_inputs(dataset, processor, image, device)
    tokenizer = processor.tokenizer
    with torch.inference_mode():
        ids = model.generate(pixel_values=pixels, decoder_input_ids=prefix,
                             max_length=1024, num_beams=1, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return baseline_module("donut").extract_entities(text), ids.clone(), text


def predict(dataset, name, sample, image):
    model, processor, id2label, device = load_model(dataset, name)
    data = baseline_module(name)
    if name == "pick":
        if not sample["ocr"]:
            return {}
        import torch
        previous = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.backends.cudnn.flags(enabled=False):
                return data.predict_sample(image, sample["ocr"], model, processor, id2label, device)
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)
    if name in ("roberta", "bros"):
        if not sample["ocr"]:
            return {}
        return data.predict_sample(sample["ocr"], model, processor, id2label, device)
    if name == "donut":
        return donut_generate(dataset, model, processor, image, device)[0]
    if not sample["ocr"]:
        return {}
    return data.predict_sample(sample["ocr"], model, processor, id2label, device, image=image)
