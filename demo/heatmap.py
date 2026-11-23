"""Fixed-target occlusion heatmaps and input-coordinate helpers."""


import json
import re
from collections import OrderedDict
from copy import deepcopy
import hashlib
import os


VISION_MODELS = {"pick", "layoutlmv3", "donut"}
_CACHE = OrderedDict()
_ALGORITHM = "blur-occlusion-20261121-v3"


def _disk_key(key):
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def choose_field(fields, requested):
    if requested and requested not in fields:
        raise ValueError("Model chưa trích xuất trường này trên ảnh. Hãy chọn một trường đã có kết quả.")
    if requested:
        return requested
    for preferred in ("total", "total.total_price", "Total_value", "company", "menu.nm", "Store_name_value"):
        if fields.get(preferred):
            return preferred
    return next((key for key, value in fields.items() if value), None)


def pick_inputs(data, image, ocr, chars, id2tag, device):
    import numpy as np
    import torch
    from PIL import Image

    helper = object.__new__(data.PickDataset)
    helper.max_boxes, helper.max_text_len, helper.image_size = 256, 64, (512, 512)
    helper.char2id = chars
    helper.tag2id = {tag: index for index, tag in id2tag.items()}
    # Predicting an uploaded receipt does not need its ground truth.
    clean = [dict(item, label="O") for item in ocr]
    boxes, texts, ids, lengths, _ = helper._process_ocr(clean)
    resized = image.resize((512, 512), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    count = len(boxes)
    mask = torch.ones(1, count, 64, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[0, i, :length] = False
    return {
        "images": ((pixels - mean) / std).unsqueeze(0).to(device),
        "boxes": torch.tensor([boxes], dtype=torch.float32, device=device),
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": mask.to(device),
        "box_mask": torch.ones(1, count, dtype=torch.bool, device=device),
        "relations": torch.from_numpy(data.compute_relations(boxes, texts, 512, 512)).unsqueeze(0).to(device),
        "return_preds": True,
    }


def generated_span(tokenizer, sequence, field):
    """Select the generated token positions overlapping the chosen JSON value."""
    ids = sequence[0].tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    match = re.search(re.escape(json.dumps(field, ensure_ascii=False)) + r"\s*:\s*", text)
    if not match:
        raise ValueError("Không tìm được trường trong JSON model đã sinh.")
    _, length = json.JSONDecoder().raw_decode(text[match.end():])
    begin, end = match.end(), match.end() + length
    positions = []
    previous = ""
    for index in range(1, len(ids)):
        current = tokenizer.decode(ids[:index + 1], skip_special_tokens=True)
        if len(current) > begin and len(previous) < end and len(current) > len(previous):
            positions.append(index - 1)
        previous = current
    if not positions:
        raise ValueError("Không tìm được token đáp án cho trường đã chọn.")
    return positions


def unpad_donut(annotation, processor, image):
    """Map encoder coordinates back through the processor's padding and rotation."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    heat = annotation["heatmap"]
    source = torch.tensor(heat["values"]).reshape(1, 1, heat["height"], heat["width"])
    canvas = functional.interpolate(source, (1280, 960), mode="bilinear", align_corners=False)[0, 0]
    unpadded = processor(image, do_pad=False, do_resize=True, size={"height": 1280, "width": 960},
                         return_tensors="pt")["pixel_values"]
    h, w = unpadded.shape[-2:]
    top, left = (1280 - h) // 2, (960 - w) // 2
    crop = canvas[top:top + h, left:left + w].numpy()
    rotated = bool(processor.do_align_long_axis and image.width > image.height)
    if rotated:
        crop = np.rot90(crop, 1).copy()
    target_h = max(1, round(64 * min(1, image.height / image.width)))
    target_w = max(1, round(64 * min(1, image.width / image.height)))
    resized = functional.interpolate(torch.from_numpy(crop.copy())[None, None],
                                     (target_h, target_w), mode="bilinear", align_corners=False)[0, 0]
    image_max = float(resized.max())
    valid = heat.get("has_signal", True) and image_max > 0
    resized = resized / image_max if valid else torch.zeros_like(resized)
    heat.update(width=target_w, height=target_h, values=resized.flatten().tolist(),
                has_signal=valid, padding_removed=True, rotation_undone=rotated)
    return annotation


def predict_with_heatmap(dataset, model_name, sample, image, field=None, scope="joint"):
    import torch
    from demo.assets import ROOT, manifest

    if model_name not in VISION_MODELS:
        raise ValueError("Chọn PICK, LayoutLMv3 hoặc Donut để xem heatmap.")
    if scope not in {"joint", "image"}:
        raise ValueError("Chọn dữ liệu heatmap hợp lệ.")
    if model_name == "donut":
        scope = "image"
    ocr = [{"text": item["text"], "box": item["box"]} for item in sample.get("ocr", [])]
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    key = (_ALGORITHM, os.environ.get("RECONIA_DEVICE", "cpu"), dataset, model_name,
           field, scope, image.size, digest, json.dumps(ocr, sort_keys=True),
           manifest()["models"][model_name]["datasets"][dataset]["checkpoint_sha256"])
    cache_file = ROOT / ".cache/demo-occlusion-heatmaps" / (_disk_key(key) + ".json")
    if key not in _CACHE and cache_file.is_file():
        try:
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            heat = saved[1]["heatmap"]
            if (heat["method"] == "Occlusion" and heat["scope"] == scope
                    and len(heat["values"]) == heat["width"] * heat["height"]
                    and all(0 <= value <= 1.000001 for value in heat["values"])):
                _CACHE[key] = saved
        except (ValueError, KeyError, TypeError, IndexError, OSError):
            pass
    if key in _CACHE:
        result = deepcopy(_CACHE[key])
        _CACHE.move_to_end(key)
        result[1]["heatmap"]["cached"] = True
        return result
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        from demo.occlusion import explain
        result = explain(dataset, model_name, sample, image, field, scope)
        result[1]["heatmap"]["cached"] = False
        result[1]["heatmap"]["algorithm_version"] = _ALGORITHM
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pending = cache_file.with_suffix(f".{os.getpid()}.tmp")
        try:
            pending.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            pending.replace(cache_file)
        except OSError:
            pending.unlink(missing_ok=True)
        _CACHE[key] = deepcopy(result)
        if len(_CACHE) > 16:
            _CACHE.popitem(last=False)
        return result
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
