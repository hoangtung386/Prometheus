from __future__ import annotations

import torch

from prometheus.losses import (
    BCEWithLogitsLoss,
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    MulticlassCombinedLoss,
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


def test_multi_class_dice_loss_ignores_absent_foreground() -> None:
    loss_fn = MultiClassDiceLoss()
    logits = torch.full((1, 3, 8, 8), -10.0)
    logits[:, 0] = 10.0
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.item() == 0.0


def test_multi_class_dice_loss_penalizes_missed_present_class() -> None:
    loss_fn = MultiClassDiceLoss()
    logits = torch.full((1, 3, 8, 8), -10.0)
    logits[:, 0] = 10.0
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    targets[:, :2, :2] = 1
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.9


def test_multiclass_combined_loss() -> None:
    loss_fn = MulticlassCombinedLoss(ce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(4, 3, 64, 64)
    targets = torch.randint(0, 3, (4, 64, 64))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_multiclass_combined_loss_perfect_match() -> None:
    loss_fn = MulticlassCombinedLoss(ce_weight=1.0, dice_weight=1.0)
    logits = torch.full((2, 3, 16, 16), -10.0)
    logits[:, 1, :, :] = 10.0
    targets = torch.ones(2, 16, 16, dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.5, f"Expected near-zero loss, got {loss.item()}"


def test_multiclass_combined_loss_gradient() -> None:
    loss_fn = MulticlassCombinedLoss()
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_multiclass_combined_loss_accepts_class_weights() -> None:
    weights = torch.tensor([0.1, 2.0, 3.0])
    loss_fn = MulticlassCombinedLoss(class_weights=weights)
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None


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
