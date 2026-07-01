from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from ..utils import GRN, LayerNorm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        has_upsample: bool = False,
        in_dim: int | None = None,
        drop_path: float = 0.0,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.has_upsample = has_upsample
        self.use_skip = use_skip
        if has_upsample:
            self.upsample = nn.ConvTranspose2d(in_dim, dim, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv2d(2 * dim, dim, kernel_size=1) if use_skip else nn.Identity()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.has_upsample:
            x = self.upsample(x)
        if self.use_skip:
            if skip is None:
                raise ValueError("Decoder block requires a skip tensor")
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.skip_proj(torch.cat([x, skip], dim=1))

        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x
