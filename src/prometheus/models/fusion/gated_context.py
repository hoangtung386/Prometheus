"""Same-grid tissue-to-nuclei context modulation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedContextFusion(nn.Module):
    def __init__(self, nuclei_channels: int, context_channels: int) -> None:
        super().__init__()
        self.context_projection = nn.Conv2d(context_channels, nuclei_channels, kernel_size=1)
        self.gate = nn.Conv2d(nuclei_channels * 2, nuclei_channels, kernel_size=1)

    def forward(self, nuclei: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = F.interpolate(context, size=nuclei.shape[-2:], mode="bilinear", align_corners=False)
        context = self.context_projection(context)
        gate = torch.sigmoid(self.gate(torch.cat([nuclei, context], dim=1)))
        return nuclei + gate * context, gate.mean()
