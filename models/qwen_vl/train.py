import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from models.qwen_vl.dataset import QwenDataset, QwenCollator
from utils.generation import train_generation
from utils.training import run_cli


def build_model(weights_dir, device):
    processor = AutoProcessor.from_pretrained(weights_dir, min_pixels=256 * 28 * 28,
                                              max_pixels=768 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        weights_dir, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa")
    language_targets = [name for name, module in model.named_modules()
                        if isinstance(module, torch.nn.Linear) and "visual" not in name
                        and name.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                           target_modules=language_targets, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)
    return model, processor


def train_model(dataset_name, epochs=30, batch_size=1, accum_steps=8, lr=1e-4,
                min_epochs=5, patience=5, resume=False, test_only=False):
    return train_generation("qwen_vl", QwenDataset, QwenCollator, build_model, dataset_name,
                            epochs=epochs, batch_size=batch_size, accum_steps=accum_steps,
                            lr=lr, min_epochs=min_epochs, patience=patience,
                            resume=resume, test_only=test_only)


if __name__ == "__main__":
    run_cli(train_model, 1, 8, 1e-4)
