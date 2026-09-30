"""Small helpers: hashing, DVC pointer parsing, git SHA, dataset version detection."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - matches DVC's content hash, not used for security
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dvc_md5(dvc_file: Path) -> str | None:
    """Return the md5 recorded in a .dvc pointer file, or None if missing."""
    if not dvc_file.exists():
        return None
    data = yaml.safe_load(dvc_file.read_text())
    return data["outs"][0]["md5"]


def git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def detect_dataset_version(md5: str | None) -> str:
    """Map a dataset md5 to v1/v2 by regenerating both deterministically in memory."""
    if not md5:
        return "unknown"
    import hashlib as _h

    from src.make_dataset import VERSIONS, csv_bytes

    for version in VERSIONS:
        if _h.md5(csv_bytes(version)).hexdigest() == md5:  # noqa: S324
            return version
    return "unknown"


def _main(argv: list[str]) -> int:
    from src.config import ROOT, load_config, path_for

    cmd = argv[0] if argv else "info"
    if cmd == "md5":
        print(md5_file(Path(argv[1])))
    elif cmd == "detect":
        print(detect_dataset_version(argv[1]))
    elif cmd == "info":
        cfg = load_config()
        raw = path_for(cfg, "raw_data", ROOT)
        dvc_md5 = read_dvc_md5(path_for(cfg, "raw_dvc", ROOT))
        print(f"Pointer (wine.csv.dvc) md5 : {dvc_md5}")
        print(f"Pointer dataset version    : {detect_dataset_version(dvc_md5)}")
        if raw.exists():
            actual = md5_file(raw)
            print(f"Working file md5           : {actual}")
            print(f"Working file version       : {detect_dataset_version(actual)}")
        else:
            print("Working file               : missing (run `make pull`)")
    else:
        print(f"unknown command {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
