"""Build an allowlisted Kaggle upload bundle; never include credentials or weights."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "docile"
    assert {s: v["documents"] for s, v in manifest["splits"].items()} == {"train": 4680, "val": 500, "test": 500}
    for name, expected in manifest["checksums"].items():
        path = (args.export / name).resolve()
        assert path.is_relative_to(args.export.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    sources = [p for p in (ROOT / "app").rglob("*")
               if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts]
    sources += [ROOT / "docs/EVALUATION.md", ROOT / "docs/KAGGLE.md"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    provenance = {"base_commit": revision, "files_sha256": {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }}
    # Kaggle expands .zip and .gz recursively. A .bin suffix preserves exact bytes.
    with zipfile.ZipFile(args.output / "reconia-code.bin", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("source-provenance.json", json.dumps(provenance, indent=2))
    with zipfile.ZipFile(args.output / "docile-export.bin", "w", zipfile.ZIP_STORED) as archive:
        archive.write(args.export / "manifest.json", "manifest.json")
        for name in manifest["checksums"]:
            archive.write(args.export / name, name)
    metadata = {
        "title": "Reconia DocILE GNN Training Bundle",
        "id": f"{args.owner}/reconia-docile-gnn-training-bundle",
        "licenses": [{"name": "other"}],
        "description": "Private research input for Reconia. Derived DocILE OCR and labels: https://docile.rossum.ai/ and https://github.com/rossumai/docile. Original data rights remain with their authors; this upload does not relicense DocILE. Local split: 4680 train, 500 val, 500 test (not official hidden test). Includes Reconia source with file hashes. No page images, credentials, or trained weights.",
    }
    (args.output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({p.name: p.stat().st_size for p in args.output.iterdir() if p.is_file()}, indent=2))


if __name__ == "__main__":
    main()
