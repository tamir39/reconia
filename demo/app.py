"""Local HTTP demo. Launch with the workspace run.bat."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import socket
import threading
from time import perf_counter
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen
import webbrowser

from demo.assets import DATASETS, MODELS, NAMES, ROOT, checkpoint, labels, manifest, samples, sample_image
from demo.heatmap import VISION_MODELS, predict_with_heatmap
from demo.inference import predict, read_image_text

PAGE = Path(__file__).with_name("index.html")
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
API_VERSION = 6
INFERENCE_LOCK = threading.Lock()


class DemoServer(ThreadingHTTPServer):
    allow_reuse_address = not hasattr(socket, "SO_EXCLUSIVEADDRUSE")
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def init_catalog(server):
    server.cases = []
    server.examples = {}
    indices = {}
    for dataset in DATASETS:
        server.examples[dataset] = {}
        for name in MODELS:
            path = Path(__file__).with_name("examples.json")
            examples = json.loads(path.read_text(encoding="utf-8"))[dataset][name]
            server.examples[dataset][name] = {}
            for kind, example in examples.items():
                sample = samples(dataset, name)[example["id"]]
                key = dataset, name, example["id"]
                if key not in indices:
                    indices[key] = len(server.cases)
                    server.cases.append({
                        "dataset": dataset, "model": name, "id": example["id"],
                        "gt": sample["gt"], "boxes": len(sample["ocr"]),
                        "image": sample_image(dataset, sample),
                    })
                server.examples[dataset][name][kind] = dict(example, case_index=indices[key])


def run_prediction(dataset, name, sample, image, heatmap, field=None, scope="joint"):
    start = perf_counter()
    with INFERENCE_LOCK:
        if heatmap and name in VISION_MODELS:
            fields, annotation = predict_with_heatmap(dataset, name, sample, image, field, scope)
        else:
            fields, annotation = predict(dataset, name, sample, image), None
    return {
        "fields": fields, "annotation": annotation, "seconds": round(perf_counter() - start, 2),
        "model": name, "dataset": dataset,
        "checkpoint_sha256": manifest()["models"][name]["datasets"][dataset]["checkpoint_sha256"],
    }


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def send_json(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/catalog":
            metadata = manifest()
            self.send_json({
                "api_version": API_VERSION, "datasets": DATASETS, "models": NAMES,
                "scores": {m: [metadata["models"][m]["datasets"][d]["metrics"]["f1"] for d in DATASETS] for m in MODELS},
                "sources": metadata, "metric": "Micro-F1",
                "dataset_fields": {d: sorted({s[2:] for s in labels(d, "layoutlmv3") if s.startswith("B-")}) for d in DATASETS},
                "supports_image_upload": True, "vision_models": sorted(VISION_MODELS),
                "ready": {d: {m: checkpoint(d, m).is_file() for m in MODELS} for d in DATASETS},
                "error_examples": {d: {m: self.server.examples[d][m]["error"] for m in MODELS} for d in DATASETS},
                "correct_examples": {d: {m: self.server.examples[d][m]["correct"] for m in MODELS} for d in DATASETS},
                "cases": [{k: v for k, v in row.items() if k != "image"} for row in self.server.cases],
            })
        elif path.startswith("/image/") and path[7:].isdigit():
            index = int(path[7:])
            if not 0 <= index < len(self.server.cases):
                self.send_error(404)
                return
            image = self.server.cases[index]["image"]
            import mimetypes
            self.send_bytes(image.read_bytes(), mimetypes.guess_type(image)[0] or "application/octet-stream")
        else:
            self.send_error(404)

    def read_length(self, maximum):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= maximum:
            raise ValueError("Dung lượng yêu cầu không hợp lệ.")
        return length

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path == "/api/predict-image":
            self.predict_image(parse_qs(url.query))
            return
        if url.path != "/api/predict":
            self.send_error(404)
            return
        try:
            payload = json.loads(self.rfile.read(self.read_length(4096)))
            index, name = payload["case"], payload["model"]
            if type(index) is not int or not 0 <= index < len(self.server.cases) or name not in MODELS:
                raise ValueError("Chọn mẫu hóa đơn và mô hình hợp lệ.")
            row = self.server.cases[index]
            if row["model"] != name:
                raise ValueError("Mẫu không thuộc mô hình đang chọn.")
            if payload.get("sample_id") != row["id"] or payload.get("dataset") != row["dataset"]:
                raise ValueError("Mẫu demo đã cập nhật. Nhấn F5 rồi chạy lại.")
            from PIL import Image
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            result = run_prediction(row["dataset"], name, samples(row["dataset"], name)[row["id"]],
                                    image, bool(payload.get("heatmap")), payload.get("field") or None,
                                    payload.get("scope", "joint"))
            result["sample_id"] = row["id"]
            self.send_json(result)
        except (ValueError, TypeError, KeyError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.log_error("Test inference: %s: %s", type(error).__name__, error)
            self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    def predict_image(self, query):
        from PIL import Image, ImageOps, UnidentifiedImageError

        dataset, name = query.get("dataset", [""])[0], query.get("model", [""])[0]
        if dataset not in DATASETS or name not in MODELS:
            self.send_json({"error": "Chọn dataset và mô hình hợp lệ."}, 400)
            return
        try:
            length = self.read_length(MAX_IMAGE_BYTES)
        except ValueError:
            self.send_json({"error": "Chọn ảnh có dung lượng tối đa 12 MB."}, 413)
            return
        try:
            with Image.open(BytesIO(self.rfile.read(length))) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Ảnh vượt quá 40 megapixel.")
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
            self.send_json({"error": str(error) if isinstance(error, ValueError) else "Không đọc được ảnh hợp lệ."}, 400)
            return
        try:
            with INFERENCE_LOCK:
                ocr = [] if name == "donut" else read_image_text(image)
            if name != "donut" and not ocr:
                self.send_json({"error": "OCR chưa đọc được chữ. Thử ảnh rõ nét hơn."}, 422)
                return
            result = run_prediction(dataset, name, {"id": "upload", "ocr": ocr}, image,
                                    query.get("heatmap", ["0"])[0] == "1", query.get("field", [""])[0] or None,
                                    query.get("scope", ["joint"])[0])
            result["ocr_boxes"] = None if name == "donut" else len(ocr)
            self.send_json(result)
        except ValueError as error:
            self.send_json({"error": str(error)}, 422)
        except Exception as error:
            self.log_error("Upload inference: %s: %s", type(error).__name__, error)
            self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="Reconia receipt extraction demo")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    try:
        server = DemoServer(("127.0.0.1", args.port), Handler)
    except OSError:
        if not args.open:
            raise
        with urlopen(f"{url}/api/catalog", timeout=3) as response:
            existing = json.load(response)
        if existing.get("api_version") != API_VERSION or existing.get("sources", {}).get("checked_at") != manifest()["checked_at"]:
            parser.exit(1, "Close the old Reconia demo window, then reopen run.bat.\n")
        webbrowser.open(url)
        return
    with server:
        init_catalog(server)
        print(f"Reconia demo: {url} ({len(server.cases)} verified examples)", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
