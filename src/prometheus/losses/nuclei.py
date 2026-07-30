"""Losses for center-based nuclei instance detection."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ..data.targets import CenterPointTargets

__all__ = ["center_focal_loss", "nuclei_regression_losses"]


def center_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CornerNet/CenterNet penalty-reduced focal loss on a Gaussian center heatmap.

    Only the exact peak counts as positive; the Gaussian skirt around it is a *discounted*
    negative via ``(1 - target) ** beta``, so a near miss is penalised far less than a
    detection in empty tissue. Normalised by the positive count so the loss does not scale
    with nucleus density, which varies by an order of magnitude across PUMA regions.
    """
    probabilities = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positives = target.eq(1).float()
    negatives = target.lt(1).float()
    positive_loss = -probabilities.log() * (1 - probabilities).pow(alpha) * positives
    negative_loss = -(1 - probabilities).log() * probabilities.pow(alpha) * (1 - target).pow(beta) * negatives
    return (positive_loss.sum() + negative_loss.sum()) / positives.sum().clamp_min(1)


def _gather(feature_map: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(C, H, W)`` features at flat spatial ``indices``, returning ``(N, C)``."""
    channels = feature_map.shape[0]
    return feature_map.reshape(channels, -1).transpose(0, 1)[indices]


def nuclei_regression_losses(
    class_logits: torch.Tensor,
    offset_map: torch.Tensor,
    size_map: torch.Tensor,
    targets: CenterPointTargets,
    class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(class, offset, size)`` losses evaluated only at ground-truth centers.

    Sampling at the annotated centers rather than densely is what makes classification
    independent of the class-agnostic detector: the classifier never sees, and is never
    penalised on, background cells.
    """
    class_losses, offset_losses, size_losses = [], [], []
    for batch_index, indices in enumerate(targets.indices):
        if indices.numel() == 0:
            continue
        class_losses.append(
            F.cross_entropy(
                _gather(class_logits[batch_index], indices),
                targets.labels[batch_index],
                weight=class_weight,
            )
        )
        offset_losses.append(F.l1_loss(_gather(offset_map[batch_index], indices), targets.offsets[batch_index]))
        size_losses.append(F.l1_loss(_gather(size_map[batch_index], indices), targets.sizes[batch_index]))

    zero = class_logits.sum() * 0.0
    return (
        torch.stack(class_losses).mean() if class_losses else zero,
        torch.stack(offset_losses).mean() if offset_losses else zero,
        torch.stack(size_losses).mean() if size_losses else zero,
    )
