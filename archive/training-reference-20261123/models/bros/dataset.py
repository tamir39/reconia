from utils.token_data import TokenDataset, TokenCollator, decode_predictions


class BrosDataset(TokenDataset):
    def __init__(self, data_file, tokenizer, label2id, max_length=512):
        super().__init__(data_file, tokenizer, label2id, max_length, layout="bros")


class BrosDataCollator(TokenCollator):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, layout="bros")
