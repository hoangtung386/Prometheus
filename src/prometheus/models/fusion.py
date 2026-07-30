"""Same-grid tissue-to-nuclei context modulation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["GatedContextFusion"]


class GatedContextFusion(nn.Module):
    """Add gated tissue-decoder context to the nuclei feature map.

    The gate is learned and its mean is returned as a diagnostic: if it collapses towards
    zero, the nuclei branch has decided tissue context is not useful, which is the signal
    that the multitask coupling is not paying for itself.
    """

    def __init__(self, nuclei_channels: int, context_channels: int) -> None:
        super().__init__()
        self.context_projection = nn.Conv2d(context_channels, nuclei_channels, kernel_size=1)
        self.gate = nn.Conv2d(nuclei_channels * 2, nuclei_channels, kernel_size=1)

    def forward(self, nuclei: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fused_features, mean_gate_activation)``."""
        resampled = F.interpolate(context, size=nuclei.shape[-2:], mode="bilinear", align_corners=False)
        projected = self.context_projection(resampled)
        gate = torch.sigmoid(self.gate(torch.cat([nuclei, projected], dim=1)))
        return nuclei + gate * projected, gate.mean()
