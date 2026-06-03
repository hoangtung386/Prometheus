from __future__ import annotations

import torch

from prometheus.losses import (
    BCEWithLogitsLoss,
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    MultiClassDiceLoss,
    TverskyLoss,
)


def test_bce_with_logits_loss() -> None:
    loss_fn = BCEWithLogitsLoss()
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_dice_loss() -> None:
    loss_fn = DiceLoss()
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_dice_loss_perfect_match() -> None:
    loss_fn = DiceLoss()
    logits = torch.full((2, 1, 16, 16), 100.0)
    targets = torch.ones(2, 1, 16, 16)
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01


def test_dice_loss_complete_mismatch() -> None:
    loss_fn = DiceLoss()
    logits = torch.full((2, 1, 16, 16), -100.0)
    targets = torch.ones(2, 1, 16, 16)
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.99


def test_focal_loss() -> None:
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_combined_loss() -> None:
    loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_multi_class_dice_loss() -> None:
    loss_fn = MultiClassDiceLoss()
    logits = torch.randn(4, 3, 64, 64)
    targets = torch.randint(0, 3, (4, 64, 64))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_tversky_loss() -> None:
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_gradient_flow() -> None:
    loss_fn = CombinedLoss()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0
