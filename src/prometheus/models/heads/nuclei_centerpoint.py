"""High-resolution center-based nuclei detection head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import FeaturePyramid

__all__ = ["NucleiCenterPointHead"]

_MAX_GROUPS = 32
_FOREGROUND_PRIOR = 0.1
_CENTER_BIAS_INIT = -math.log((1 - _FOREGROUND_PRIOR) / _FOREGROUND_PRIOR)


class NucleiCenterPointHead(nn.Module):
    """Predict class-agnostic centers plus per-cell class, sub-pixel offset and size.

    Detection is deliberately class-agnostic and the taxonomy is a separate classifier,
    following the transfer boundary CellViT++ exploits: a broadly pretrained cell detector
    stays stable while a small classifier adapts to the target label set. A class-specific
    center map plus a class head is internally inconsistent - one nucleus produces one peak
    per class.

    Args:
        dims: Encoder channel widths per stage.
        num_classes: Number of nucleus classes for the classification head.
        output_stride: 4 or 8. Stride 4 fuses the s4 and s8 pyramid levels; stride 8 uses
            s8 alone. Geometry is always predicted at high resolution and never
            reconstructed from the stride-32 semantic bottleneck.
    """

    def __init__(self, dims: list[int], num_classes: int, output_stride: int = 4) -> None:
        super().__init__()
        if output_stride not in {4, 8}:
            raise ValueError(f"output_stride must be 4 or 8, got {output_stride}")
        self.output_stride = output_stride
        feature_dim = dims[0] if output_stride == 4 else dims[1]
        groups = math.gcd(_MAX_GROUPS, feature_dim)

        self.s8_projection = nn.Conv2d(dims[1], feature_dim, kernel_size=1)
        self.s4_projection = nn.Conv2d(dims[0], feature_dim, kernel_size=1) if output_stride == 4 else nn.Identity()
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, feature_dim),
            nn.SiLU(),
        )
        self.center = nn.Conv2d(feature_dim, 1, kernel_size=1)
        self.classes = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.offsets = nn.Conv2d(feature_dim, 2, kernel_size=1)
        self.sizes = nn.Conv2d(feature_dim, 2, kernel_size=1)

        # Bias the center head towards a low foreground prior. Without this, the first
        # steps predict ~0.5 everywhere and the focal loss is dominated by the vast
        # negative area, which stalls the head.
        if self.center.bias is None:  # pragma: no cover - Conv2d always allocates a bias
            raise RuntimeError("The center head must have a bias to initialise")
        nn.init.constant_(self.center.bias, _CENTER_BIAS_INIT)

    def build_feature(self, features: FeaturePyramid) -> torch.Tensor:
        """Fuse pyramid levels into the single map every nuclei head reads from."""
        s8 = self.s8_projection(features.s8)
        if self.output_stride == 8:
            return self.refinement(s8)
        s8 = F.interpolate(s8, size=features.s4.shape[-2:], mode="bilinear", align_corners=False)
        return self.refinement(self.s4_projection(features.s4) + s8)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(center_logits, class_logits, offsets, sizes)``; sizes are non-negative."""
        return (
            self.center(feature),
            self.classes(feature),
            self.offsets(feature),
            F.softplus(self.sizes(feature)),
        )
