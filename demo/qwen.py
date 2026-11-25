"""Load Thinh's Qwen adapters on the matching pretrained model."""
import importlib.util
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from demo.assets import ROOT, assert_checkpoint, labels, model_assets

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
BASE_DIR = ROOT / "weights" / "qwen_vl"
MAX_NEW_TOKENS = 1024


def base_manifest():
    return json.loads((model_assets("qwen_vl") / "base_model.json").read_text(encoding="utf-8"))


def download_base():
    """Download the pinned base in resumable ranges and verify each file."""
    import requests

    manifest = base_manifest()
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    def download(item):
        name, size = item["name"], item["bytes"]
        target = (BASE_DIR / name).resolve()
        if not target.is_relative_to(BASE_DIR.resolve()):
            raise ValueError("Đường dẫn tài nguyên Qwen không hợp lệ.")

        def valid(path):
            if not path.is_file() or path.stat().st_size != size:
                return False
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]

        if valid(target):
            print("Verified Qwen base:", name, flush=True)
            return
        temporary = target.with_suffix(target.suffix + ".part")
        start = temporary.stat().st_size if temporary.exists() else 0
        if start > size or (0 < start < size <= 32 * 1024 * 1024):
            temporary.unlink()
            start = 0
        url = f"https://huggingface.co/{BASE_MODEL}/resolve/{BASE_REVISION}/{name}"
        try:
            with requests.Session() as session:
                while start < size:
                    end = min(start + 32 * 1024 * 1024, size) - 1
                    headers = {"Range": f"bytes={start}-{end}"} if size > 32 * 1024 * 1024 else {}
                    with session.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                        if headers:
                            expected = f"bytes {start}-{end}/{size}"
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                                raise ValueError("Qwen server trả về đoạn tải không hợp lệ: " + name)
                        elif response.status_code != 200:
                            raise ValueError("Không tải được tài nguyên Qwen: " + name)
                        written = 0
                        with temporary.open("ab") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                stream.write(chunk)
                                written += len(chunk)
                        if written != end - start + 1:
                            raise ValueError("Tài nguyên Qwen tải chưa đủ: " + name)
                    start += written
                    if size > 32 * 1024 * 1024:
                        print(f"Qwen {name}: {start / 1_000_000:.0f}/{size / 1_000_000:.0f} MB", flush=True)
        except requests.RequestException:
            raise RuntimeError("Tải Qwen bị gián đoạn. Chạy lại lệnh để tiếp tục: " + name) from None
        if not valid(temporary):
            temporary.unlink(missing_ok=True)
            raise ValueError("Tài nguyên Qwen không khớp bản gốc: " + name)
        temporary.replace(target)
        print("Downloaded and verified Qwen base:", name, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, manifest["files"]))


def runtime_status():
    for package in ("peft", "accelerate", "bitsandbytes"):
        if importlib.util.find_spec(package) is None:
            return False, "Chạy setup.bat để cài thư viện Qwen."
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                "tokenizer.json", "chat_template.json", "model.safetensors.index.json")
    if not all((BASE_DIR / name).is_file() for name in required):
        return False, "Tải Qwen: python -m demo.fetch_assets --model qwen_vl --base"
    try:
        for item in base_manifest()["files"]:
            path = BASE_DIR / item["name"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                return False, "Qwen tải chưa đủ. Chạy lại lệnh tải mô hình nền."
        index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
        size = 0
        for name in set(index["weight_map"].values()):
            path = (BASE_DIR / name).resolve()
            if not path.is_relative_to(BASE_DIR.resolve()) or not path.is_file():
                return False, "Qwen tải chưa đủ. Chạy lại lệnh tải mô hình nền."
            size += path.stat().st_size
        if size < index["metadata"]["total_size"]:
            return False, "Qwen tải chưa đủ. Chạy lại lệnh tải mô hình nền."
    except (ValueError, KeyError, OSError):
        return False, "Cấu hình Qwen chưa đầy đủ. Chạy lại lệnh tải mô hình nền."
    return True, ""


def load_adapter(model, dataset):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    state = torch.load(assert_checkpoint(dataset, "qwen_vl"), map_location="cpu", weights_only=True, mmap=True)
    expected = get_peft_model_state_dict(model)
    if state.keys() != expected.keys() or any(state[key].shape != expected[key].shape for key in state):
        raise ValueError("Adapter Qwen không khớp mô hình nền.")
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys:
        raise ValueError("Không nạp được đầy đủ adapter Qwen.")
    model.requires_grad_(False)
    model.eval()


def load(dataset, device):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    ready, reason = runtime_status()
    if not ready:
        raise ValueError(reason)
    processor = AutoProcessor.from_pretrained(BASE_DIR, local_files_only=True, use_fast=True)
    processor.image_processor.size = {"shortest_edge": 256 * 28 * 28, "longest_edge": 512 * 28 * 28}
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        llm_int8_skip_modules=["model.visual", "lm_head"],
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_DIR, local_files_only=True, quantization_config=quantization,
        torch_dtype=torch.float16, device_map={"": str(device)},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    # Match the frozen base/vision precision used by the published trainer.
    # This disables parameter updates; no training or gradient checkpointing runs.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.model.visual.to(torch.float32)
    model.model.visual.forward = torch.amp.custom_fwd(
        device_type=device.type, cast_inputs=torch.float32
    )(model.model.visual.forward)
    config = LoraConfig.from_pretrained(model_assets("qwen_vl"))
    model = get_peft_model(model, config)
    load_adapter(model, dataset)
    return model, processor, dict(enumerate(labels(dataset))), device


def predict(dataset, model, processor, device, image):
    from models.qwen_vl.dataset import predict_sample

    return predict_sample(image, model, processor, device, labels=labels(dataset),
                          max_new_tokens=MAX_NEW_TOKENS)
