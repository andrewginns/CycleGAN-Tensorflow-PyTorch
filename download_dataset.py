#!/usr/bin/env python3
"""Download and extract CycleGAN datasets.

This replaces the legacy `wget + unzip` flow so it works in minimal
containers and avoids depending on `unzip` being installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "ae_photos",
    "apple2orange",
    "summer2winter_yosemite",
    "horse2zebra",
    "monet2photo",
    "cezanne2photo",
    "ukiyoe2photo",
    "vangogh2photo",
    "maps",
    "cityscapes",
    "facades",
    "iphone2dslr_flower",
}


BASE_URLS = [
    # Original URL from the historic script (often 404 now)
    "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/",
    # Commonly used mirror for the same datasets
    "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/",
]


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help=f"One of: {', '.join(sorted(DATASETS))}")
    parser.add_argument("--root", default="datasets", help="Dataset root directory")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract")
    args = parser.parse_args(argv)

    dataset = args.dataset
    if dataset not in DATASETS:
        print(f"Unknown dataset: {dataset}", file=sys.stderr)
        print(f"Available datasets are: {', '.join(sorted(DATASETS))}", file=sys.stderr)
        return 2

    root = Path(args.root)
    target_dir = root / dataset
    zip_path = root / f"{dataset}.zip"

    if target_dir.exists() and not args.force:
        print(f"Already present: {target_dir}")
        return 0

    if target_dir.exists() and args.force:
        shutil.rmtree(target_dir)

    last_error: Exception | None = None
    for base in BASE_URLS:
        url = f"{base}{dataset}.zip"
        try:
            print(f"Downloading: {url}")
            _download(url, zip_path)
            break
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            last_error = e
            continue
    else:
        print("Failed to download dataset from all known URLs.", file=sys.stderr)
        if last_error is not None:
            print(f"Last error: {last_error}", file=sys.stderr)
        return 1

    print(f"Extracting: {zip_path} -> {root}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)

    zip_path.unlink(missing_ok=True)
    print(f"Done: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
