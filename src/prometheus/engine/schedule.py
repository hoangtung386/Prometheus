"""Learning-rate schedule and optimizer parameter grouping."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..models import PrometheusNet

__all__ = ["build_optimizer", "warmup_cosine_lambda"]


def warmup_cosine_lambda(
    warmup_epochs: int,
    total_epochs: int,
    minimum_ratio: float,
) -> Callable[[int], float]:
    """Return a ``LambdaLR`` multiplier: linear warmup, then cosine decay to ``minimum_ratio``.

    Stepped **per epoch**. With patch-based training the step count per epoch grows by an
    order of magnitude and a per-iteration schedule becomes the better choice; see
    ``docs/phan-tich-tissue-va-ke-hoach.md`` section 3.1.
    """
    if warmup_epochs < 0 or total_epochs <= 0:
        raise ValueError("warmup_epochs must be non-negative and total_epochs positive")

    def schedule(epoch: int) -> float:
        """Return the learning-rate multiplier for ``epoch``."""
        if epoch < warmup_epochs:
            return max(minimum_ratio, (epoch + 1) / max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return schedule


def build_optimizer(
    model: PrometheusNet,
    lr: float,
    backbone_lr_multiplier: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.Optimizer:
    """Build AdamW with the pretrained encoder on a reduced learning rate.

    The encoder starts from ConvNeXt-V2 FCMAE/ImageNet-22K weights while the decoders and
    heads start from scratch. Training both at one rate either destroys the pretrained
    features or starves the heads, so they get separate parameter groups.
    """
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    task_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": lr * backbone_lr_multiplier},
            {"params": task_parameters, "lr": lr},
        ],
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
