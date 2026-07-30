"""Multi-class tissue segmentation losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["MultiClassDiceLoss", "MulticlassCombinedLoss"]


class MultiClassDiceLoss(nn.Module):
    """Soft Dice over class channels, pooled across the batch.

    Pooling across the batch (rather than averaging per-image Dice) matches the official
    micro Dice and keeps the gradient of a rare class usable when only one image in the
    batch contains it.

    Args:
        smooth: Numerical stabiliser added to numerator and denominator.
        include_background: Whether channel 0 contributes. The challenge does not score
            ``tissue_white_background``, so this stays ``False``.
        ignore_absent: Drop classes that are absent from the whole batch. Leave this
            ``False`` whenever the sampler guarantees rare-class presence, otherwise a
            hallucinated rare class goes unpunished on batches that lack it.
        class_weights: Per-class weights, renormalised to mean 1 over the active classes.
    """

    class_weights: torch.Tensor | None

    def __init__(
        self,
        smooth: float = 1e-6,
        include_background: bool = False,
        ignore_absent: bool = False,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.ignore_absent = ignore_absent
        self.register_buffer(
            "class_weights",
            None if class_weights is None else class_weights.detach().float(),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dims)
        cardinality = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
        loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        active = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
        if not self.include_background and num_classes > 1:
            active[0] = False
        if self.ignore_absent:
            active &= one_hot.sum(dim=dims) > 0
        if not active.any():
            return logits.sum() * 0.0

        loss = loss[active]
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)[active]
            loss = loss * (weights / weights.mean().clamp_min(self.smooth))
        return loss.mean()


class MulticlassCombinedLoss(nn.Module):
    """Weighted sum of cross-entropy and soft Dice, exposing both components for logging."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
        smooth: float = 1e-6,
        include_background_dice: bool = False,
        ignore_absent_dice: bool = False,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MultiClassDiceLoss(
            smooth=smooth,
            include_background=include_background_dice,
            ignore_absent=ignore_absent_dice,
            class_weights=class_weights,
        )

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the unweighted ``(cross_entropy, dice)`` terms."""
        return self.cross_entropy(logits, targets), self.dice(logits, targets)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy, dice = self.components(logits, targets)
        return self.ce_weight * cross_entropy + self.dice_weight * dice
