"""Exponential moving average of model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn

__all__ = ["WeightEma"]


class WeightEma:
    """Track a smoothed copy of ``model``'s weights.

    On a ~20 image validation fold, per-epoch scores are noisy enough that checkpoint
    selection partly samples that noise. Selecting on an averaged model both reduces the
    variance and generalises slightly better, so ``select_inference_state`` prefers the EMA
    weights while ``model_state`` stays the live weights an exact resume needs.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend ``model``'s parameters into the average and copy its buffers verbatim.

        Buffers are copied rather than averaged: they hold counters and running statistics
        whose averaged value is not meaningful.
        """
        for averaged, current in zip(self.model.parameters(), model.parameters(), strict=True):
            averaged.mul_(self.decay).add_(current.detach(), alpha=1.0 - self.decay)
        for averaged_buffer, current_buffer in zip(self.model.buffers(), model.buffers(), strict=True):
            averaged_buffer.copy_(current_buffer)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the averaged weights, for storage under ``ema_state``."""
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore previously stored averaged weights."""
        self.model.load_state_dict(state)
