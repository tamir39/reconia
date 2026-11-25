"""Download published checkpoints without starting a training job."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def target_path(item):
    base = Path(os.environ.get("RECONIA_CHECKPOINT_DIR", ROOT / "checkpoints")).resolve()
    relative = Path(item["local"]).relative_to("checkpoints")
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Checkpoint target is outside its directory")
    return target


def verified(item, target):
    return target.is_file() and target.stat().st_size == item["bytes"] and sha(target) == item["sha256"]


def fetch(items):
    pending = []
    for item in items:
        target = target_path(item)
        if verified(item, target):
            print("Verified:", item["local"])
        else:
            pending.append((item, target))
    if not pending:
        return
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    api = KaggleApi()
    api.authenticate()  # Use the user's existing Kaggle configuration; never print it.
    with api.build_kaggle_client() as client:
        for item, target in pending:
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug, request.kernel_slug = item["kernel"].split("/")
            request.version_number = item["version"]
            request.file_path = item["remote"]
            try:
                download = client.kernels.kernels_api_client.download_kernel_output(request)
            except Exception:
                raise RuntimeError(f"Cannot access {item['kernel']} version {item['version']}: {item['remote']}") from None
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            try:
                with requests.get(download.url, stream=True, timeout=(30, 180)) as response:
                    if response.status_code != 200:
                        raise RuntimeError("Cannot download " + item["remote"])
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            stream.write(chunk)
                            digest.update(chunk)
                if temporary.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                    raise RuntimeError("Published output changed; pinned hash differs: " + item["remote"])
                temporary.replace(target)
                print("Downloaded and verified:", item["local"])
            except requests.RequestException:
                raise RuntimeError("Download failed: " + item["remote"]) from None
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("roberta", "bros", "pick", "layoutlmv3", "donut", "qwen_vl", "qwen"))
    parser.add_argument("--dataset", choices=("sroie", "cord_v2", "wildreceipt"))
    parser.add_argument("--list", action="store_true", help="Show checkpoint sources without downloading")
    args = parser.parse_args()
    model = "qwen_vl" if args.model == "qwen" else args.model
    manifest = json.loads((ROOT / "demo/data/checkpoints.json").read_text(encoding="utf-8"))
    items = [item for item in manifest["files"] if item["kind"] == "checkpoint" and
             (not model or item["model"] == model) and
             (not args.dataset or item["dataset"] == args.dataset)]
    if not items:
        raise ValueError("No checkpoints match the selection")
    if args.list:
        for item in items:
            print(f"{item['model']}/{item['dataset']} | {item['kernel']} v{item['version']} | {item['bytes'] / 1_000_000:.0f} MB")
        return
    fetch(items)
    for item in items:
        if not item.get("adapter_config"):
            continue
        source = ROOT / item["adapter_config"]
        target = target_path(item).with_name("adapter_config.json")
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != json.loads(source.read_text(encoding="utf-8")):
            raise ValueError("Existing adapter configuration differs: " + item["dataset"])
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    # The six published models share one label inventory per dataset.
    for dataset in sorted({item["dataset"] for item in items}):
        source = ROOT / "demo/data/labels" / f"{dataset}.json"
        base = Path(os.environ.get("RECONIA_CHECKPOINT_DIR", ROOT / "checkpoints"))
        target = base / dataset / "labels.json"
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != json.loads(source.read_text(encoding="utf-8")):
            raise ValueError("Existing labels differ from the pinned final inventory: " + dataset)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
