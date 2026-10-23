from utils.token_data import TokenDataset, TokenCollator, decode_predictions


class LayoutLMv3Dataset(TokenDataset):
    def __init__(self, data_file, processor, label2id, data_dir, max_length=512):
        processor.image_processor.apply_ocr = False
        super().__init__(data_file, processor.tokenizer, label2id, max_length,
                         layout="layoutlmv3", image_processor=processor.image_processor,
                         data_dir=data_dir)


class LayoutLMv3Collator(TokenCollator):
    pass
