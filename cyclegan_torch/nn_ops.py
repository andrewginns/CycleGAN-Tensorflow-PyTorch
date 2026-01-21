from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def same_padding_2d(
    input_height: int,
    input_width: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int, int, int]:
    """Return (left, right, top, bottom) padding to mimic TF SAME."""

    if isinstance(kernel_size, int):
        kernel_height = kernel_width = kernel_size
    else:
        kernel_height, kernel_width = kernel_size

    if isinstance(stride, int):
        stride_height = stride_width = stride
    else:
        stride_height, stride_width = stride

    if isinstance(dilation, int):
        dilation_height = dilation_width = dilation
    else:
        dilation_height, dilation_width = dilation

    out_height = math.ceil(input_height / stride_height)
    out_width = math.ceil(input_width / stride_width)

    pad_height = max(
        0,
        (out_height - 1) * stride_height + (kernel_height - 1) * dilation_height + 1 - input_height,
    )
    pad_width = max(
        0,
        (out_width - 1) * stride_width + (kernel_width - 1) * dilation_width + 1 - input_width,
    )

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


class Conv2dSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_left, pad_right, pad_top, pad_bottom = same_padding_2d(
            input_height=x.shape[-2],
            input_width=x.shape[-1],
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        if pad_left or pad_right or pad_top or pad_bottom:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class ConvTranspose2dSame(nn.Module):
    """ConvTranspose2d configured to match TF SAME for k=3, s=2 used in this repo."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size != 3 or stride != 2:
            raise ValueError("Only kernel_size=3, stride=2 is supported for SAME transpose conv")
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)
