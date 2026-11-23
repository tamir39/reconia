"""Fixed-target, input-space occlusion for receipt explanations.

Each cell is measured by an actual model forward. OCR can be masked with its
image region; the original prediction target and geometry remain fixed.
"""
import math
from PIL import ImageFilter

from demo.inference import baseline_module, donut_generate, donut_inputs, load_model


def crf_log_marginals(emissions, crf):
    """Forward/backward log marginals for one unpadded PICK character sequence."""
    import torch

    emissions = emissions.double()
    transitions = crf.transitions.double()
    alpha = [crf.start_transitions.double() + emissions[0]]
    for emission in emissions[1:]:
        alpha.append(torch.logsumexp(alpha[-1][:, None] + transitions, dim=0) + emission)
    beta = [None] * len(emissions)
    beta[-1] = crf.end_transitions.double()
    for i in range(len(emissions) - 2, -1, -1):
        beta[i] = torch.logsumexp(transitions + emissions[i + 1][None, :] + beta[i + 1][None, :], dim=1)
    normalizer = torch.logsumexp(alpha[-1] + crf.end_transitions.double(), dim=0)
    return torch.stack(alpha) + torch.stack(beta) - normalizer


def blurred_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=max(image.size) / 24))


def measure(inputs, pixel_key, reference, evaluate, base_score, ocr, mask_words, grid, scope):
    """Evaluate every cell, rather than using a gradient to rank candidate cells."""
    import torch

    pixels = inputs[pixel_key]
    height, width = pixels.shape[-2:]
    rows, cols = grid
    cells, drops = [], []
    repeated = float(evaluate(inputs))
    jitter = abs(base_score - repeated)
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for col in range(cols):
            x0, x1 = col * width // cols, (col + 1) * width // cols
            changed = dict(inputs)
            altered = pixels.clone()
            altered[:, :, y0:y1, x0:x1] = reference[:, :, y0:y1, x0:x1]
            changed[pixel_key] = altered
            selected = []
            if scope == "joint":
                for index, word in enumerate(ocr):
                    left, top, right, bottom = word["box"]
                    cx, cy = (left + right) / 2000, (top + bottom) / 2000
                    if col / cols <= cx < (col + 1) / cols and row / rows <= cy < (row + 1) / rows:
                        selected.append(index)
                changed = mask_words(changed, selected)
            after = float(evaluate(changed))
            drop = base_score - after
            if not math.isfinite(drop):
                raise ValueError("Không tính được mức ảnh hưởng của vùng ảnh.")
            drops.append(drop)
            cells.append({"row": row, "col": col, "box": [1000 * col / cols, 1000 * row / rows,
                           1000 * (col + 1) / cols, 1000 * (row + 1) / rows],
                          "masked_words": selected, "score_after": after, "score_drop": drop})
    threshold = max(1e-4, jitter * 10)
    positive = torch.tensor([value if value > threshold else 0 for value in drops])
    maximum = float(positive.max())
    positive_effect = maximum > 0
    valid = maximum >= max(threshold, .005)
    values = positive / maximum if positive_effect else positive
    return {
        "method": "Occlusion", "scope": scope, "score": "mean_log_probability",
        "width": cols, "height": rows, "values": values.tolist(), "has_signal": valid,
        "reference_score": base_score, "max_score_drop": maximum, "noise_floor": threshold,
        "repeat_error": jitter, "strength": min(1.0, maximum / .05),
        "baseline": "gaussian_blur", "cells": cells,
        "positive_effect_present": positive_effect, "display_floor": .005,
        "cell_coordinate_space": "model_input", "model_input_size": [width, height],
    }


def layout_context(dataset, image, sample, requested, scope):
    import torch
    from demo.heatmap import choose_field

    model, processor, id2label, device = load_model(dataset, "layoutlmv3")
    data = baseline_module("layoutlmv3")
    ocr = sample["ocr"]
    if not ocr:
        raise ValueError("Ảnh chưa có chữ để giải thích dự đoán.")
    encoding = processor(image, [o["text"] for o in ocr], boxes=[o["box"] for o in ocr],
                         max_length=512, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in encoding.items()}
    logits = model(**inputs).logits
    pred = logits[0].argmax(-1).tolist()
    word_ids = encoding.word_ids()
    word_labels = data.align_labels(pred, word_ids, id2label)
    fields = data.extract_entities(ocr, word_labels)
    field = choose_field(fields, requested)
    if not field:
        raise ValueError("Model chưa trích xuất được trường nào để tạo heatmap.")
    positions, seen = [], set()
    for position, word in enumerate(word_ids):
        if word is not None and word not in seen:
            if word_labels.get(word, "O")[2:] == field:
                positions.append(position)
            seen.add(word)
    tags = torch.tensor([pred[p] for p in positions], device=device)
    def score(output):
        return output[0, positions].float().log_softmax(-1)[torch.arange(len(positions), device=device), tags].mean()
    def evaluate(changed):
        return score(model(**changed).logits)
    def mask_words(changed, selected):
        changed["input_ids"] = changed["input_ids"].clone()
        tokens = [i for i, word in enumerate(word_ids) if word in selected]
        changed["input_ids"][0, tokens] = processor.tokenizer.mask_token_id
        return changed
    reference = processor.image_processor(blurred_image(image), return_tensors="pt")["pixel_values"].to(device)
    regions = [{"kind": "predicted", "box": item["box"]} for i, item in enumerate(ocr)
               if word_labels.get(i, "O")[2:] == field]
    return fields, field, inputs, reference, evaluate, float(score(logits)), ocr, mask_words, regions, "pixel_values"


def pick_context(dataset, image, sample, requested, scope):
    import torch
    from demo.heatmap import choose_field, pick_inputs

    model, chars, id2tag, device = load_model(dataset, "pick")
    data = baseline_module("pick")
    ocr = sample["ocr"][:256]
    if not ocr:
        raise ValueError("Ảnh chưa có chữ để giải thích dự đoán.")
    inputs = pick_inputs(data, image, ocr, chars, id2tag, device)
    capture = {}
    handle = model.decoder.register_forward_hook(lambda module, args, output: capture.update(emissions=output))
    transformer = model.encoder.transformer_encoder
    original_forward = transformer.forward
    text_outputs = {}
    text_hook = transformer.register_forward_hook(lambda module, args, output: text_outputs.update(last=output.clone()))
    active = {"words": []}
    class Handles:
        def remove(self):
            transformer.forward = original_forward
            text_hook.remove()
            handle.remove()
    cleanup = Handles()
    try:
        _, predictions = model(**inputs)
        original = capture["emissions"].clone()
        original_text = text_outputs["last"]
        predictions = predictions[0]
        tags = data.align_labels(predictions, id2tag)
        fields = data.extract_entities(ocr, tags)
        field = choose_field(fields, requested)
        if not field:
            raise ValueError("Model chưa trích xuất được trường nào để tạo heatmap.")
        selected = {box: [(char, tag) for char, tag in enumerate(sequence) if id2tag[tag][2:] == field]
                    for box, sequence in enumerate(predictions)}
        selected = {box: values for box, values in selected.items() if values}
        target_boxes = list(selected)
        compact_indices = {box: i for i, box in enumerate(target_boxes)}
        def score(emissions, compact=False):
            scores = []
            for box, target in selected.items():
                index = compact_indices[box] if compact else box
                marginal = crf_log_marginals(emissions[0, index, :len(predictions[box])], model.crf)
                scores.extend(marginal[char, tag] for char, tag in target)
            return torch.stack(scores).mean()
        def evaluate(changed):
            active["words"] = (changed["input_ids"][0] != inputs["input_ids"][0]).any(-1).nonzero().flatten().tolist()
            fused, nodes = model.encoder(changed["images"], changed["boxes"], changed["input_ids"], changed["attention_mask"])
            graph, _ = model.glcn(nodes, changed["box_mask"], changed["relations"])
            # BiLSTM/CRF sequences are independent across boxes after graph fusion.
            # Evaluate only target-box emissions, preserving every context node.
            emissions = model.decoder(fused[:, target_boxes], graph[:, target_boxes])
            return score(emissions, compact=True)
        def mask_words(changed, indices):
            changed["input_ids"] = changed["input_ids"].clone()
            for index in indices:
                length = len(predictions[index])
                changed["input_ids"][0, index, :length] = chars["<unk>"]
            return changed
        reference = pick_inputs(data, blurred_image(image), ocr, chars, id2tag, device)["images"]
        regions = [{"kind": "predicted", "box": item["box"]} for i, item in enumerate(ocr) if tags.get(i, "O")[2:] == field]
        # The character transformer processes boxes independently before graph fusion.
        # Cache both possible texts for each box; image, fusion, graph and decoder
        # still run for every perturbation. No weights or predictions are changed.
        all_unknown = mask_words(dict(inputs), list(range(len(predictions))))
        model.encoder(all_unknown["images"], all_unknown["boxes"], all_unknown["input_ids"], all_unknown["attention_mask"])
        unknown_text = text_outputs["last"]
        text_hook.remove()
        def cached_text(src, *args, **kwargs):
            output = original_text.clone()
            if active["words"]:
                output[active["words"]] = unknown_text[active["words"]]
            return output
        transformer.forward = cached_text
        # Check a mixed group against the full uncached computation, not just
        # the all-original/all-unknown endpoints used to construct the cache.
        probe = mask_words(dict(inputs), list(range(0, len(predictions), 3)))
        cached_score = float(evaluate(probe))
        transformer.forward = original_forward
        model(**dict(probe, return_preds=False))
        uncached_score = float(score(capture["emissions"]))
        if abs(cached_score - uncached_score) > 1e-5:
            raise ValueError("PICK không khớp giữa hai đường tính heatmap.")
        transformer.forward = cached_text
        return (fields, field, inputs, reference, evaluate, float(score(original)), ocr, mask_words, regions, "images"), cleanup
    except BaseException:
        cleanup.remove()
        raise


def donut_context(dataset, image, requested):
    import torch
    from demo.heatmap import choose_field, generated_span

    model, processor, _, device = load_model(dataset, "donut")
    fields, sequence, _ = donut_generate(dataset, model, processor, image, device)
    field = choose_field(fields, requested)
    if not field:
        raise ValueError("Donut chưa sinh được trường hợp lệ để tạo heatmap.")
    positions = generated_span(processor.tokenizer, sequence, field)
    # Avoid scoring the unused suffix of the generated JSON.
    end = max(positions) + 1
    decoder_ids = sequence[:, :end]
    targets = sequence[0, [p + 1 for p in positions]]
    pixels, _ = donut_inputs(dataset, processor, image, device)
    inputs = {"pixel_values": pixels, "decoder_input_ids": decoder_ids, "use_cache": False}
    def evaluate(changed):
        logits = model(**changed).logits[0, positions].float().log_softmax(-1)
        return logits.gather(1, targets[:, None]).mean()
    reference, _ = donut_inputs(dataset, processor, blurred_image(image), device)
    return fields, field, inputs, reference, evaluate, float(evaluate(inputs)), [], None, [], "pixel_values"


def explain(dataset, model_name, sample, image, requested=None, scope="joint"):
    import torch
    from demo.heatmap import unpad_donut

    if scope not in {"joint", "image"}:
        raise ValueError("Chọn dữ liệu heatmap hợp lệ.")
    handle = None
    with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False):
        try:
            if model_name == "layoutlmv3":
                context = layout_context(dataset, image, sample, requested, scope)
            elif model_name == "pick":
                context, handle = pick_context(dataset, image, sample, requested, scope)
            else:
                scope = "image"
                context = donut_context(dataset, image, requested)
            fields, field, inputs, reference, evaluate, score, ocr, mask_words, regions, pixel_key = context
            heat = measure(inputs, pixel_key, reference, evaluate, score, ocr, mask_words,
                           (5, 5) if model_name == "donut" else (7, 7), scope)
            heat.update(field=field, target_values=fields[field], target_source="model_prediction",
                        target_fixed=True, ocr_fixed=scope == "image")
            heat["target_kind"] = {"layoutlmv3": "predicted_word_tags",
                                   "pick": "predicted_character_crf_marginals",
                                   "donut": "generated_value_tokens"}[model_name]
            if model_name == "pick":
                heat["score"] = "mean_crf_log_marginal_probability"
            annotation = {"size": list(image.size), "regions": regions, "heatmap": heat}
            if model_name == "donut":
                annotation = unpad_donut(annotation, load_model(dataset, model_name)[1].image_processor, image)
            return fields, annotation
        finally:
            if handle is not None:
                handle.remove()
