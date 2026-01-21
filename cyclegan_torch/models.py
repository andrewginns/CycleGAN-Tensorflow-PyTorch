from __future__ import annotations

import torch
from torch import nn

from .nn_ops import Conv2dSame, ConvTranspose2dSame


class ResnetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(channels, channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            Conv2dSame(channels, channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GeneratorResNet9(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, dim: int = 64) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        layers += [
            Conv2dSame(in_channels, dim, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        ]

        layers += [
            Conv2dSame(dim, dim * 2, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(dim * 2, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            Conv2dSame(dim * 2, dim * 4, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(dim * 4, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(9):
            layers.append(ResnetBlock(dim * 4))

        layers += [
            ConvTranspose2dSame(dim * 4, dim * 2, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(dim * 2, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            ConvTranspose2dSame(dim * 2, dim, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        ]

        layers += [
            Conv2dSame(dim, out_channels, kernel_size=7, stride=1, bias=True),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 64) -> None:
        super().__init__()

        def conv(
            in_ch: int,
            out_ch: int,
            stride: int,
            norm: bool,
        ) -> nn.Sequential:
            layers: list[nn.Module] = [
                Conv2dSame(in_ch, out_ch, kernel_size=4, stride=stride, bias=not norm)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            conv(in_channels, dim, stride=2, norm=False),
            conv(dim, dim * 2, stride=2, norm=True),
            conv(dim * 2, dim * 4, stride=2, norm=True),
            conv(dim * 4, dim * 8, stride=1, norm=True),
            Conv2dSame(dim * 8, 1, kernel_size=4, stride=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
