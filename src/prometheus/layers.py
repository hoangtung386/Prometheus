"""Low-level neural network primitives shared by the encoder, decoders and heads.

These are faithful ConvNeXt-V2 building blocks (``GRN``, no layer scale), which is what
makes the one-to-one ``timm`` weight mapping in
:mod:`prometheus.models.backbones.pretrained` possible. Renaming their parameters or
changing their shapes breaks pretrained loading and every existing checkpoint.
"""

from __future__ import annotations

from typing import Literal

import torch
from timm.layers import DropPath
from torch import nn
from torch.nn import functional as F

__all__ = ["GRN", "ConvNeXtBlock", "LayerNorm"]

DataFormat = Literal["channels_last", "channels_first"]


class LayerNorm(nn.Module):
    """Layer normalization over ``(N, H, W, C)`` or ``(N, C, H, W)`` tensors."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: DataFormat = "channels_last",
    ) -> None:
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"Unsupported data_format: {data_format!r}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(inputs, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = inputs.mean(1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(1, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-V2), channels-last."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        global_norm = torch.norm(inputs, p=2, dim=(1, 2), keepdim=True)
        normalized = global_norm / (global_norm.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (inputs * normalized) + self.beta + inputs


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-V2 residual block: 7x7 depthwise conv, LayerNorm, then MLP with GRN."""

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.norm(self.dwconv(inputs).permute(0, 2, 3, 1))
        residual = self.pwconv2(self.grn(self.act(self.pwconv1(residual))))
        return inputs + self.drop_path(residual.permute(0, 3, 1, 2))
