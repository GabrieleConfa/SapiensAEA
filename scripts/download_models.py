"""Download the MediaPipe Hand Landmarker model and track its hash.

    python -m scripts.download_models              # download (first time, or check for changes)
    python -m scripts.download_models --verify     # only hash the local file
    python -m scripts.download_models --accept-new # replace the local model if 'latest' changed

The URL points to "latest", so the model can change under the same name. Every download is logged
in models/MODELS.md with its SHA-256 and date. If the hash differs from the last recorded one,
the script warns and keeps the local model unless --accept-new is given: landmarks from different
model versions should not be mixed within the study. The .task file itself is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

from common.config import REPO_ROOT, load_config

MODELS_MD = REPO_ROOT / "models" / "MODELS.md"
HEADER = """# Models

Downloaded models are git-ignored; this file records what was downloaded and when, so results can
be tied to an exact model. Rows are appended by `python -m scripts.download_models`.

| file | date | sha256 | bytes | status | url |
|------|------|--------|-------|--------|-----|
"""
_ROW = re.compile(r"^\|\s*(?P<file>[^|]+?)\s*\|\s*(?P<date>[^|]+?)\s*\|\s*(?P<sha>[0-9a-f]{64})\s*\|"
                  r"\s*(?P<size>[^|]*?)\s*\|\s*(?P<status>[^|]*?)\s*\|")
NOT_INSTALLED = "CHANGED, not installed"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_hashes(file_name: str) -> list[tuple[str, str]]:
    """(date, sha256) of the models actually installed for file_name, oldest first.

    Rows of downloads that were refused ("CHANGED, not installed") are skipped: the reference is
    always the model in use.
    """
    if not MODELS_MD.exists():
        return []
    rows = []
    for line in MODELS_MD.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m and m["file"] == file_name and m["status"] != NOT_INSTALLED:
            rows.append((m["date"], m["sha"]))
    return rows


def append_row(file_name: str, sha: str, size: int, status: str, url: str) -> None:
    if not MODELS_MD.exists():
        MODELS_MD.parent.mkdir(parents=True, exist_ok=True)
        MODELS_MD.write_text(HEADER, encoding="utf-8", newline="\n")
    with open(MODELS_MD, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"| {file_name} | {date.today().isoformat()} | {sha} | {size} | {status} | {url} |\n")


def download(url: str, dest: Path) -> Path:
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    return part


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="hash the local model, no download")
    ap.add_argument("--accept-new", action="store_true", help="replace the local model if the hash changed")
    args = ap.parse_args(argv)
    cfg = load_config()
    url = cfg["mediapipe.model_url"]
    dest = REPO_ROOT / cfg["mediapipe.model_path"]
    history = recorded_hashes(dest.name)
    last = history[-1][1] if history else None

    if args.verify:
        if not dest.exists():
            print(f"missing {dest}; run without --verify to download it")
            return 1
        sha = sha256_of(dest)
        ok = sha == last
        print(f"{dest.name}: {sha} ({'matches' if ok else 'DOES NOT MATCH'} the last recorded hash {last})")
        return 0 if ok else 1

    part = download(url, dest)
    sha = sha256_of(part)
    size = part.stat().st_size
    print(f"sha256 {sha}  ({size} bytes)")

    if last is None:
        part.replace(dest)
        append_row(dest.name, sha, size, "first download", url)
        print(f"saved {dest} and recorded it in {MODELS_MD}")
        return 0
    if sha == last:
        part.replace(dest)
        append_row(dest.name, sha, size, "unchanged", url)
        print("same model as the last recorded download")
        return 0

    print("\n" + "!" * 78)
    print(f"WARNING: the 'latest' model has CHANGED.\n  recorded: {last} ({history[-1][0]})\n  new:      {sha}")
    print("Landmarks from different model versions should not be mixed within the study.")
    print("!" * 78 + "\n")
    if dest.exists() and not args.accept_new:
        part.unlink()
        append_row(dest.name, sha, size, NOT_INSTALLED, url)
        print(f"kept the local model {dest}; rerun with --accept-new to replace it")
        return 2
    part.replace(dest)
    append_row(dest.name, sha, size, "CHANGED, installed", url)
    print(f"installed the new model at {dest}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
