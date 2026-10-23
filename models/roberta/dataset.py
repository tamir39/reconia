import torch

from utils.token_data import TokenDataset, decode_predictions


class RobertaDataset(TokenDataset):
    pass


def predict_sample(ocr, model, tokenizer, id2label, device, max_length=512):
    from utils.token_data import encode_document

    sample = {"id": "inference", "ocr": ocr}
    predictions = []
    model.eval()
    with torch.no_grad():
        for ids, mask, _, _ in encode_document(sample, tokenizer, max_length):
            output = model(input_ids=torch.tensor([ids], device=device),
                           attention_mask=torch.tensor([mask], device=device))
            predictions.append(output.logits.argmax(-1)[0].cpu().tolist())
    return decode_predictions(predictions, [sample], tokenizer, id2label, max_length)[0]
