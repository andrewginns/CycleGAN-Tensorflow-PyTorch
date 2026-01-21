#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cyclegan_torch.scripts.infer import _device, _load_generator


def _preprocess(load_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((load_size, load_size), interpolation=Image.BICUBIC),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt file")
    parser.add_argument("--direction", choices=["a2b", "b2a"], default="a2b")
    parser.add_argument("--dataset", required=True, help="path of images folder")
    parser.add_argument(
        "--load_size",
        type=int,
        default=286,
        help="resize images to this square size before crop",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="center crop to this square size before inference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit number of images (0 = all)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of warmup images (excluded from timing)",
    )
    args = parser.parse_args()

    image_paths = sorted(glob(args.dataset.rstrip("/") + "/*.jpg"))
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise SystemExit(f"No .jpg files found in {args.dataset}")

    device = _device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    net = _load_generator(Path(args.ckpt), args.direction, device)
    tfm = _preprocess(args.load_size, args.crop_size)

    def load_one(path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = tfm(img).unsqueeze(0)
            return x

    # Warmup (model load + first kernel launches).
    for p in image_paths[: max(args.warmup, 0)]:
        x = load_one(p).to(device)
        with torch.no_grad():
            _ = net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start = time.time()
    first_100_end = None

    with torch.no_grad():
        for i, p in enumerate(image_paths):
            x = load_one(p).to(device)
            _ = net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

            if i == 99:
                first_100_end = time.time()

    end_all = time.time()

    total = len(image_paths)
    all_s = end_all - start

    print(f"Device: {device.type}")
    print(f"Images: {total}")

    if first_100_end is not None:
        s_100 = first_100_end - start
        print(f"First 100: {s_100:.4f}s ({100 / s_100:.4f} fps)")

    print(f"All: {all_s:.4f}s ({total / all_s:.4f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
