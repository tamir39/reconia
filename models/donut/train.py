from transformers import DonutProcessor, VisionEncoderDecoderModel

from models.donut.dataset import DonutDataset, DonutCollator
from utils.generation import train_generation
from utils.training import run_cli


def build_model(weights_dir, device):
    processor = DonutProcessor.from_pretrained(weights_dir)
    processor.image_processor.size = {"height": 1280, "width": 960}
    model = VisionEncoderDecoderModel.from_pretrained(weights_dir)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.generation_config.forced_eos_token_id = None
    model.decoder.gradient_checkpointing_enable()
    model.config.decoder.use_cache = False
    return model.to(device), processor


def train_model(dataset_name, epochs=30, batch_size=1, accum_steps=8, lr=3e-5,
                min_epochs=5, patience=5, resume=False, test_only=False):
    return train_generation("donut", DonutDataset, DonutCollator, build_model, dataset_name,
                            epochs=epochs, batch_size=batch_size, accum_steps=accum_steps,
                            lr=lr, min_epochs=min_epochs, patience=patience,
                            resume=resume, test_only=test_only, max_new_tokens=1535)


if __name__ == "__main__":
    run_cli(train_model, 1, 8, 3e-5)
