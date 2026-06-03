from __future__ import annotations

import numpy.random as random

import torch
import torch.nn as nn

try:
    from MinkowskiEngine import (
        SparseTensor,
        MinkowskiConvolution,
        MinkowskiLinear,
        MinkowskiDepthwiseConvolution,
        to_sparse,
    )
    _ME_AVAILABLE = True
except ImportError:
    SparseTensor = None
    MinkowskiConvolution = None
    MinkowskiLinear = None
    MinkowskiDepthwiseConvolution = None
    to_sparse = None
    _ME_AVAILABLE = False


class MinkowskiEngineNotAvailableError(ImportError):
    def __init__(self) -> None:
        super().__init__(
            "MinkowskiEngine is required but not installed. "
            "Install it via: pip install MinkowskiEngine"
        )


def require_minkowski_engine() -> None:
    if not _ME_AVAILABLE:
        raise MinkowskiEngineNotAvailableError


class MinkowskiGRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        require_minkowski_engine()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
            self.gamma * (x.F * Nx) + self.beta + x.F,
            coordinate_map_key=in_key,
            coordinate_manager=cm)


class MinkowskiDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True) -> None:
        super().__init__()
        require_minkowski_engine()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
            x.F * mask,
            coordinate_map_key=in_key,
            coordinate_manager=cm)


class MinkowskiLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        require_minkowski_engine()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)
