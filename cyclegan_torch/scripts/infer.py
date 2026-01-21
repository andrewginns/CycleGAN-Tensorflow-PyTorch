from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from ..models import GeneratorResNet9


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_generator(ckpt_path: Path, direction: str, device: torch.device) -> GeneratorResNet9:
    ckpt = torch.load(ckpt_path, map_location=device)
    if direction == "a2b":
        state = ckpt["g_a2b"]
    elif direction == "b2a":
        state = ckpt["g_b2a"]
    else:
        raise ValueError("direction must be a2b or b2a")

    net = GeneratorResNet9().to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def _preprocess(load_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((load_size, load_size), interpolation=Image.BICUBIC),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def _postprocess(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config file (optional)")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt file")
    parser.add_argument("--direction", choices=["a2b", "b2a"], default="a2b")
    parser.add_argument("--input", required=True, help="Input image or folder")
    parser.add_argument("--output", required=True, help="Output image or folder")
    parser.add_argument("--load_size", type=int, default=286)
    parser.add_argument("--crop_size", type=int, default=256)
    args = parser.parse_args(argv)

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if isinstance(cfg, dict):
            args.load_size = int(cfg.get("load_size", args.load_size))
            args.crop_size = int(cfg.get("crop_size", args.crop_size))

    device = _device()

    ckpt_path = Path(args.ckpt)
    net = _load_generator(ckpt_path, args.direction, device)

    in_path = Path(args.input)
    out_path = Path(args.output)

    tfm = _preprocess(args.load_size, args.crop_size)

    def run_one(src: Path, dst: Path) -> None:
        with Image.open(src) as img:
            img = img.convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                y = net(x)[0]
            out_img = _postprocess(y)
            dst.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(dst)

    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png"}
        for p in sorted(in_path.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            run_one(p, out_path / p.name)
    else:
        run_one(in_path, out_path)

    return 0
