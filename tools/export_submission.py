"""Package Git-listed source and small demo/benchmark assets, excluding local data."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".cache", ".venv", "venv", "data", "weights", "checkpoints", "results", "output", "dist"}


def allowed(name):
    path = PurePosixPath(name)
    return (path.parts[0] not in EXCLUDED and "__pycache__" not in path.parts
            and not any(part == ".env" or part.startswith(".env.") for part in path.parts)
            and path.suffix.lower() not in {".pt", ".pth", ".safetensors", ".lnk", ".zip", ".pyc"})


def export(output):
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")
    names = sorted(name for name in tracked if name and allowed(name) and (ROOT / name).is_file())
    missing = [name for name in ("run.bat", "setup.bat", "SUBMISSION.md", "demo/app.py", "demo/samples.json") if name not in names]
    if missing:
        raise RuntimeError("Stage the final source before packaging; missing: " + ", ".join(missing))
    manifest = {
        "branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
        "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_changes": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            content = (ROOT / name).read_bytes()
            manifest["files"][name] = hashlib.sha256(content).hexdigest()
            archive.writestr("reconia/" + name, content)
        archive.writestr("reconia/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"Exported {len(names)} files: {output} ({output.stat().st_size / 1048576:.1f} MiB)")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/reconia-final-source.zip")
    export(parser.parse_args().output)
