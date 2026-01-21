from __future__ import annotations

import torch
from torch import nn


class LSGANLoss:
    def __init__(self) -> None:
        self.mse = nn.MSELoss()

    def d_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
        return self.mse(logits_real, torch.ones_like(logits_real)) + self.mse(
            logits_fake, torch.zeros_like(logits_fake)
        )

    def g_loss(self, logits_fake: torch.Tensor) -> torch.Tensor:
        return self.mse(logits_fake, torch.ones_like(logits_fake))


def l1_cycle_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - recon))
