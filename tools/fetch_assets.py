"""Download only missing final checkpoints/prepared splits, verifying pinned hashes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(items):
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    grouped = defaultdict(list)
    for item in items:
        target = (ROOT / item["local"]).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError("Asset target is outside the repository")
        if target.is_file() and target.stat().st_size == item["bytes"] and sha(target) == item["sha256"]:
            print("Verified:", item["local"])
        else:
            grouped[item["kernel"]].append((item, target))
    if not grouped:
        return
    api = KaggleApi()
    api.authenticate()  # Use the user's existing Kaggle configuration; never print it.
    for kernel, pending in grouped.items():
        if api.kernels_status(kernel).status.name != "COMPLETE":
            raise RuntimeError(f"{kernel} is not complete; no job is started or restarted")
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = kernel.split("/")
        request.page_size = 100
        files = {}
        with api.build_kaggle_client() as client:
            while True:
                page = client.kernels.kernels_api_client.list_kernel_session_output(request)
                files.update({entry.file_name: entry.url for entry in page.files if entry})
                if not page.next_page_token:
                    break
                request.page_token = page.next_page_token
        for item, target in pending:
            if item["remote"] not in files:
                raise RuntimeError("Pinned output is missing: " + item["remote"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            try:
                with requests.get(files[item["remote"]], stream=True, timeout=(30, 180)) as response:
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
    parser.add_argument("--checkpoints", action="store_true", help="Fetch the 15 large best checkpoints")
    parser.add_argument("--prepared", action="store_true", help="Fetch nine published JSONL data splits")
    parser.add_argument("--model", choices=("roberta", "bros", "pick", "layoutlmv3", "donut"))
    args = parser.parse_args()
    if not args.checkpoints and not args.prepared:
        parser.error("Choose --checkpoints and/or --prepared; nothing is downloaded by default")
    manifest = json.loads((ROOT / "demo/external-assets.json").read_text())
    wanted = {kind for kind, enabled in (("checkpoint", args.checkpoints), ("prepared", args.prepared)) if enabled}
    items = [item for item in manifest["files"] if item["kind"] in wanted and
             (not args.model or item["kind"] == "prepared" or f"/{args.model}/" in item["local"])]
    fetch(items)
    # All five published models use the same observed BIO inventory per dataset.
    for dataset in ("sroie", "cord_v2", "wildreceipt"):
        source = ROOT / "demo/metadata" / dataset / "layoutlmv3/labels.json"
        target = ROOT / "checkpoints" / dataset / "labels.json"
        if target.exists() and json.loads(target.read_text()) != json.loads(source.read_text()):
            raise ValueError("Existing labels differ from the pinned final inventory: " + dataset)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
