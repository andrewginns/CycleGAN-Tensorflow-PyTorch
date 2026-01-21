from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    rows: int,
    cols: int,
) -> None:
    """Save a [N,3,H,W] tensor in [-1,1] to a tiled JPEG/PNG."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu().clamp(-1, 1)
    images = (images + 1.0) * 0.5  # [0,1]
    images = (images * 255.0).to(torch.uint8)

    n, c, h, w = images.shape
    assert c == 3
    assert n == rows * cols

    grid = torch.zeros((h * rows, w * cols, 3), dtype=torch.uint8)

    idx = 0
    for r in range(rows):
        for col in range(cols):
            img = images[idx].permute(1, 2, 0)
            grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = img
            idx += 1

    Image.fromarray(grid.numpy()).save(path)
