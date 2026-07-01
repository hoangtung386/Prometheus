"""High-resolution center-based nuclei detection head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts import FeaturePyramid


class NucleiCenterPointHead(nn.Module):
    def __init__(self, dims: list[int], num_classes: int, output_stride: int = 4) -> None:
        super().__init__()
        self.output_stride = output_stride
        feature_dim = dims[0] if output_stride == 4 else dims[1]
        self.s8_projection = nn.Conv2d(dims[1], feature_dim, kernel_size=1)
        self.s4_projection = nn.Conv2d(dims[0], feature_dim, kernel_size=1) if output_stride == 4 else nn.Identity()
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(math.gcd(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(math.gcd(32, feature_dim), feature_dim),
            nn.SiLU(),
        )
        self.center = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.classes = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.offsets = nn.Conv2d(feature_dim, 2, kernel_size=1)
        self.sizes = nn.Conv2d(feature_dim, 2, kernel_size=1)
        nn.init.constant_(self.center.bias, -2.19)

    def build_feature(self, features: FeaturePyramid) -> torch.Tensor:
        s8 = self.s8_projection(features.s8)
        if self.output_stride == 8:
            return self.refinement(s8)
        s8 = F.interpolate(s8, size=features.s4.shape[-2:], mode="bilinear", align_corners=False)
        return self.refinement(self.s4_projection(features.s4) + s8)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.center(feature), self.classes(feature), self.offsets(feature), F.softplus(self.sizes(feature))
