from __future__ import annotations

import torch.nn as nn

from ..utils.minkowski_utils import (
    MinkowskiConvolution,
    MinkowskiDropPath,
    MinkowskiGRN,
    MinkowskiLayerNorm,
    MinkowskiLinear,
    require_minkowski_engine,
)


class MinkowskiConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0., D: int = 2) -> None:
        super().__init__()
        require_minkowski_engine()
        self.dwconv = MinkowskiConvolution(
            dim, dim, kernel_size=3, stride=1, groups=dim, bias=True, dimension=D
        )
        self.norm = MinkowskiLayerNorm(dim, eps=1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim, bias=True, dimension=D)
        self.act = nn.GELU()
        self.grn = MinkowskiGRN(4 * dim)
        self.pwconv2 = MinkowskiLinear(4 * dim, dim, bias=True, dimension=D)
        self.drop_path = MinkowskiDropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x
