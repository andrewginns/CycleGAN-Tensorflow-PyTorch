from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class DomainConfig:
    root: Path
    load_size: int
    crop_size: int
    train: bool
    legacy_minmax_norm: bool


def _build_transform(cfg: DomainConfig) -> transforms.Compose:
    ops: list[transforms.Transform] = []

    ops.append(transforms.Resize((cfg.load_size, cfg.load_size), interpolation=Image.BICUBIC))
    if cfg.train:
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomCrop((cfg.crop_size, cfg.crop_size)))
    else:
        ops.append(transforms.CenterCrop((cfg.crop_size, cfg.crop_size)))

    ops.append(transforms.ToTensor())  # [0,1]

    if cfg.legacy_minmax_norm:
        # Match old TF behavior: per-image min/max normalization.
        def _minmax(x: torch.Tensor) -> torch.Tensor:
            x_min = torch.min(x)
            x_max = torch.max(x)
            if (x_max - x_min) < 1e-6:
                return torch.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        ops.append(transforms.Lambda(_minmax))

    # Map [0,1] -> [-1,1]
    ops.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))

    return transforms.Compose(ops)


class ImageFolderDataset(Dataset[torch.Tensor]):
    def __init__(self, root: str | Path, load_size: int, crop_size: int, train: bool, legacy_minmax_norm: bool):
        self.cfg = DomainConfig(
            root=Path(root),
            load_size=load_size,
            crop_size=crop_size,
            train=train,
            legacy_minmax_norm=legacy_minmax_norm,
        )
        self.paths = sorted(
            [p for p in self.cfg.root.glob("*.jpg")] + [p for p in self.cfg.root.glob("*.png")]
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.cfg.root}")
        self.transform = _build_transform(self.cfg)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx % len(self.paths)]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


class UnpairedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root_a: str | Path,
        root_b: str | Path,
        load_size: int,
        crop_size: int,
        train: bool,
        legacy_minmax_norm: bool,
    ) -> None:
        self.a = ImageFolderDataset(root_a, load_size, crop_size, train, legacy_minmax_norm)
        self.b = ImageFolderDataset(root_b, load_size, crop_size, train, legacy_minmax_norm)

    def __len__(self) -> int:
        return max(len(self.a), len(self.b))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.a[idx]
        b = self.b[torch.randint(low=0, high=len(self.b), size=(1,)).item()]
        return a, b
