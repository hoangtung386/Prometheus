"""Losses for center-based nuclei instance detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.targets import CenterPointTargets


def center_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positives = target.eq(1).float()
    negatives = target.lt(1).float()
    negative_weights = (1 - target).pow(4)
    positive_loss = -(probabilities.log()) * (1 - probabilities).pow(2) * positives
    negative_loss = -(1 - probabilities).log() * probabilities.pow(2) * negative_weights * negatives
    positive_count = positives.sum()
    return (positive_loss.sum() + negative_loss.sum()) / positive_count.clamp_min(1)


def _gather_map(feature_map: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    channels = feature_map.shape[0]
    return feature_map.reshape(channels, -1).transpose(0, 1)[indices]


def nuclei_regression_losses(
    class_logits: torch.Tensor,
    offset_map: torch.Tensor,
    size_map: torch.Tensor | None,
    targets: CenterPointTargets,
    class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_losses, offset_losses, size_losses = [], [], []
    for batch_index, indices in enumerate(targets.indices):
        if indices.numel() == 0:
            continue
        class_losses.append(
            F.cross_entropy(
                _gather_map(class_logits[batch_index], indices),
                targets.labels[batch_index],
                weight=class_weight,
            )
        )
        offset_losses.append(F.l1_loss(_gather_map(offset_map[batch_index], indices), targets.offsets[batch_index]))
        if size_map is not None:
            size_losses.append(F.l1_loss(_gather_map(size_map[batch_index], indices), targets.sizes[batch_index]))
    zero = class_logits.sum() * 0.0
    class_loss = torch.stack(class_losses).mean() if class_losses else zero
    offset_loss = torch.stack(offset_losses).mean() if offset_losses else zero
    size_loss = torch.stack(size_losses).mean() if size_losses else zero
    return class_loss, offset_loss, size_loss
