#!/usr/bin/env python
"""
Download the GDSC release files listed in data/MANIFEST.json and verify their SHA-256.

Usage:
    python scripts/download_gdsc.py            # download missing files, verify all
    python scripts/download_gdsc.py --force    # re-download even if present

The manifest pins the exact release (8.5, 27 Oct 2023) used to build the committed
GBM subset, so a fresh clone reproduces the same inputs byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "MANIFEST.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_bytes: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", expected_bytes or 0))
        with (
            dest.open("wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="re-download files even if they exist")
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help=f"destination directory (default: {RAW_DIR})"
    )
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST_PATH.read_text())
    base_url = manifest["base_url"]
    failures = 0

    for filename, meta in manifest["files"].items():
        dest = args.raw_dir / filename
        if dest.exists() and not args.force:
            print(f"exists    {filename}")
        else:
            print(f"download  {filename}")
            download(base_url + filename, dest, meta.get("bytes"))

        actual = sha256sum(dest)
        if actual == meta["sha256"]:
            print(f"verified  {filename}")
        else:
            failures += 1
            print(f"MISMATCH  {filename}\n  expected {meta['sha256']}\n  actual   {actual}", file=sys.stderr)

    if failures:
        print(
            f"\n{failures} file(s) failed verification. GDSC may have re-issued the release;", file=sys.stderr
        )
        print("do not build the GBM subset from unverified inputs.", file=sys.stderr)
        return 1
    print(f"\nAll {len(manifest['files'])} files verified against GDSC release {manifest['gdsc_release']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
