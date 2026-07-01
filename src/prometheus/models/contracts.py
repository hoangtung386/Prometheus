"""Stable typed outputs for model, loss, and inference composition."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class FeaturePyramid:
    s4: torch.Tensor
    s8: torch.Tensor
    s16: torch.Tensor
    s32: torch.Tensor


@dataclass
class MultitaskOutput:
    tissue_logits: torch.Tensor
    nuclei_center_logits: torch.Tensor
    nuclei_class_logits: torch.Tensor
    nuclei_offsets: torch.Tensor
    nuclei_sizes: torch.Tensor | None = None
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)
