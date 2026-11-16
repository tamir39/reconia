import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.donut.dataset import extract_entities as extract_donut_entities
from models.layoutlmv3.dataset import align_labels as align_layoutlmv3_labels
from models.layoutlmv3.dataset import extract_entities as extract_layoutlmv3_entities
from models.pick.dataset import align_labels as align_pick_labels
from models.pick.dataset import compute_relations, extract_entities as extract_pick_entities
from models.qwen_vl.dataset import SYSTEM_PROMPT, extract_entities as extract_qwen_entities, format_prompt


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    norm_heatmap = heatmap.astype(np.float32)
    min_val, max_val = norm_heatmap.min(), norm_heatmap.max()
    if max_val > min_val:
        norm_heatmap = (norm_heatmap - min_val) / (max_val - min_val)
    else:
        norm_heatmap = np.zeros_like(norm_heatmap)

    heatmap_tensor = torch.from_numpy(norm_heatmap).unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(
        heatmap_tensor,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    resized = resized_tensor.numpy()

    r = np.clip(1.5 - np.abs(4.0 * resized - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * resized - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * resized - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    heatmap_pil = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")

    return Image.blend(image, heatmap_pil, alpha=alpha)


def visualize_layoutlmv3(
    ocr: list[dict],
    model: torch.nn.Module,
    processor,
    id2label: dict[int, str],
    device: torch.device,
    image: Image.Image,
    max_length: int = 512,
) -> tuple[dict[str, list[str]], Image.Image]:
    words = [item["text"] for item in ocr]
    boxes = [item["box"] for item in ocr]
    image_rgb = image.convert("RGB")

    encoding = processor(
        image_rgb,
        words,
        boxes=boxes,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            bbox=encoding["bbox"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
            pixel_values=encoding["pixel_values"].to(device),
            output_attentions=True,
        )

    pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    labels = align_layoutlmv3_labels(pred_ids, encoding.word_ids(), id2label)
    entities = extract_layoutlmv3_entities(ocr, labels)

    if outputs.attentions:
        last_attn = outputs.attentions[-1][0]
        mean_attn = last_attn.mean(dim=0)
        seq_len = encoding["input_ids"].shape[1]
        visual_attn = mean_attn[:seq_len, -196:]
        token_visual_weights = visual_attn.mean(dim=0)
        heatmap = token_visual_weights.view(14, 14).cpu().numpy()
    else:
        heatmap = np.zeros((14, 14), dtype=np.float32)

    overlay = overlay_heatmap(image, heatmap)
    return entities, overlay


def visualize_pick(
    ocr: list[dict],
    model: torch.nn.Module,
    char2id: dict[str, int],
    id2tag: dict[int, str],
    device: torch.device,
    image: Image.Image,
    max_boxes: int = 100,
    max_text_len: int = 64,
    image_size: tuple[int, int] = (512, 512),
) -> tuple[dict[str, list[str]], Image.Image]:
    image_rgb = image.convert("RGB")
    width, height = image_size
    resized_img = image_rgb.resize((width, height), Image.Resampling.BILINEAR)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = torch.from_numpy(np.array(resized_img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    images = ((image_tensor - mean) / std).unsqueeze(0).to(device)

    boxes: list[list[float]] = []
    texts: list[str] = []
    input_ids: list[list[int]] = []
    lengths: list[int] = []

    for item in ocr[:max_boxes]:
        x1, y1, x2, y2 = item["box"]
        x1 = max(0.0, min(float(width), x1 / 1000.0 * width))
        y1 = max(0.0, min(float(height), y1 / 1000.0 * height))
        x2 = max(0.0, min(float(width), x2 / 1000.0 * width))
        y2 = max(0.0, min(float(height), y2 / 1000.0 * height))

        if x2 <= x1:
            x2 = min(float(width), x1 + 1.0)
            x1 = min(x1, x2 - 1.0)
        if y2 <= y1:
            y2 = min(float(height), y1 + 1.0)
            y1 = min(y1, y2 - 1.0)

        boxes.append([x1, y1, x2, y2])
        text = item["text"]
        texts.append(text)

        text_len = min(len(text), max_text_len)
        pad_len = max_text_len - text_len
        lengths.append(text_len)
        char_ids = [char2id.get(c, 1) for c in text[:text_len]]
        input_ids.append(char_ids + [0] * pad_len)

    num_boxes = len(boxes)
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0).to(device)
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    box_mask = torch.ones(1, num_boxes, dtype=torch.bool, device=device)
    attention_mask = torch.ones(1, num_boxes, max_text_len, dtype=torch.bool, device=device)
    for box_idx, text_len in enumerate(lengths):
        if text_len > 0:
            attention_mask[0, box_idx, :text_len] = False

    relations = compute_relations(boxes, texts, width, height)
    relations_tensor = torch.from_numpy(relations).unsqueeze(0).to(device)

    feature_maps: list[torch.Tensor] = []

    def hook_fn(_, __, output):
        feature_maps.append(output)

    hook_handle = model.encoder.layer3.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        _, preds = model(
            images=images,
            boxes=boxes_tensor,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            box_mask=box_mask,
            relations=relations_tensor,
            return_preds=True,
        )

    hook_handle.remove()

    labels = align_pick_labels(preds[0], id2tag)
    entities = extract_pick_entities(ocr[:max_boxes], labels)

    fmap = feature_maps[0][0]
    heatmap = torch.mean(fmap, dim=0).cpu().numpy()

    overlay = overlay_heatmap(image, heatmap)
    return entities, overlay


def visualize_donut(
    image: Image.Image,
    model: torch.nn.Module,
    processor,
    device: torch.device,
    dataset_name: str,
    max_length: int = 1024,
) -> tuple[dict[str, list[str]], Image.Image]:
    image_rgb = image.convert("RGB")
    encoding = processor(
        image_rgb,
        return_tensors="pt",
        do_resize=True,
        size={"height": 1280, "width": 960},
    )
    pixel_values = encoding["pixel_values"].to(device)

    task_prompt = f"<s_{dataset_name}>"
    prompt_ids = processor.tokenizer.encode(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=prompt_ids,
            max_length=max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    decoded = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    entities = extract_donut_entities(decoded)

    if outputs.cross_attentions:
        step_maps = []
        for step in outputs.cross_attentions:
            last_layer_cross = step[-1][0].mean(dim=0)
            step_maps.append(last_layer_cross.mean(dim=0))

        mean_cross = torch.stack(step_maps, dim=0).mean(dim=0)
        num_patches = mean_cross.numel()

        if num_patches == 1200:
            heatmap = mean_cross.view(40, 30).cpu().numpy()
        else:
            side = int(num_patches ** 0.5)
            heatmap = mean_cross[:side * side].view(side, side).cpu().numpy()
    else:
        heatmap = np.zeros((40, 30), dtype=np.float32)

    overlay = overlay_heatmap(image, heatmap)
    return entities, overlay


def visualize_qwen_vl(
    image: Image.Image,
    model: torch.nn.Module,
    processor,
    device: torch.device,
    labels: list[str] | None = None,
    max_new_tokens: int = 1024,
) -> tuple[dict[str, list[str]], Image.Image]:
    prompt_text = format_prompt(labels)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if SYSTEM_PROMPT:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    prompt_chat = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    prompt_inputs = processor(
        [image],
        [prompt_chat],
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        outputs = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    prompt_len = prompt_inputs["input_ids"].shape[1]
    new_tokens = outputs.sequences[:, prompt_len:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    entities = extract_qwen_entities(decoded)

    grid = prompt_inputs["image_grid_thw"][0]
    spatial_merge = processor.image_processor.merge_size
    h_grid = int(grid[1].item()) // spatial_merge
    w_grid = int(grid[2].item()) // spatial_merge

    vision_mask = prompt_inputs["input_ids"][0] == processor.image_token_id

    if outputs.attentions and vision_mask.any():
        step_attns = []
        for step in outputs.attentions:
            last_layer_attn = step[-1][0].mean(dim=0)
            step_attns.append(last_layer_attn[-1, :prompt_len])

        attn_to_prompt = torch.stack(step_attns, dim=0).mean(dim=0)
        visual_weights = attn_to_prompt[vision_mask]
        
        if visual_weights.numel() == h_grid * w_grid:
            heatmap = visual_weights.view(h_grid, w_grid).float().cpu().numpy()
        else:
            side = int(visual_weights.numel() ** 0.5)
            heatmap = visual_weights[:side * side].view(side, side).float().cpu().numpy()
    else:
        heatmap = np.zeros((h_grid, w_grid), dtype=np.float32)

    overlay = overlay_heatmap(image, heatmap)
    return entities, overlay