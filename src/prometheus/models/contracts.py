"""Stable typed outputs for model, loss and inference composition.

Every model in this project returns a named dataclass, never a positional tuple: the
multitask output has five dense maps whose order is impossible to remember and whose
shapes are similar enough that a swap fails silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["FeaturePyramid", "MultitaskOutput"]


@dataclass(frozen=True)
class FeaturePyramid:
    """Encoder features named by their stride relative to the input image."""

    s4: torch.Tensor
    s8: torch.Tensor
    s16: torch.Tensor
    s32: torch.Tensor


@dataclass(frozen=True)
class MultitaskOutput:
    """Dense outputs of :class:`prometheus.models.PrometheusNet`.

    Attributes:
        tissue_logits: ``(B, num_tissue_classes, H, W)`` at input resolution.
        nuclei_center_logits: ``(B, 1, H/s, W/s)``. Class-agnostic by design, so the
            taxonomy is learned by ``nuclei_class_logits`` alone and a nucleus cannot
            produce one peak per class.
        nuclei_class_logits: ``(B, num_nucleus_types, H/s, W/s)``.
        nuclei_offsets: ``(B, 2, H/s, W/s)`` signed sub-pixel ``(dx, dy)`` per cell.
        nuclei_sizes: ``(B, 2, H/s, W/s)`` non-negative ``(width, height)`` in cell units.
        auxiliary: Diagnostics that are logged but never consumed downstream.
    """

    tissue_logits: torch.Tensor
    nuclei_center_logits: torch.Tensor
    nuclei_class_logits: torch.Tensor
    nuclei_offsets: torch.Tensor
    nuclei_sizes: torch.Tensor
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)
