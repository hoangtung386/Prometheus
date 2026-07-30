from __future__ import annotations

import pytest
import torch

from prometheus.losses import MulticlassCombinedLoss, MultiClassDiceLoss


def test_multi_class_dice_loss_is_scalar_and_positive() -> None:
    torch.manual_seed(0)
    loss = MultiClassDiceLoss()(torch.randn(4, 3, 64, 64), torch.randint(0, 3, (4, 64, 64)))

    assert loss.ndim == 0
    assert loss.item() > 0


def test_multi_class_dice_loss_penalizes_a_missed_present_class() -> None:
    # Class 1 is present but never predicted -> its term is ~1. Class 2 is absent from both
    # and contributes ~0, so the mean over the two foreground channels lands near 0.5.
    logits = torch.full((1, 3, 8, 8), -10.0)
    logits[:, 0] = 10.0
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    targets[:, :2, :2] = 1

    assert MultiClassDiceLoss()(logits, targets).item() == pytest.approx(0.5, abs=0.1)


def test_dice_loss_punishes_a_hallucinated_absent_class_by_default() -> None:
    # ignore_absent=False is the default so a rare class the model invents on a batch that
    # lacks it is still penalised. See docs/phan-tich-tissue-va-ke-hoach.md section 3.3.
    logits = torch.full((1, 3, 8, 8), -10.0)
    logits[:, 1] = 10.0
    targets = torch.zeros(1, 8, 8, dtype=torch.long)

    assert MultiClassDiceLoss()(logits, targets).item() > 0.4
    assert MultiClassDiceLoss(ignore_absent=True)(logits, targets).item() == 0.0


def test_background_channel_is_excluded_from_dice() -> None:
    logits = torch.full((1, 3, 8, 8), -10.0)
    logits[:, 0] = 10.0
    targets = torch.zeros(1, 8, 8, dtype=torch.long)

    assert MultiClassDiceLoss(ignore_absent=True)(logits, targets).item() == 0.0
    assert MultiClassDiceLoss(include_background=True, ignore_absent=True)(logits, targets).item() < 0.01


def test_combined_loss_reaches_zero_on_a_perfect_prediction() -> None:
    logits = torch.full((2, 3, 16, 16), -10.0)
    logits[:, 1] = 10.0
    targets = torch.ones(2, 16, 16, dtype=torch.long)

    assert MulticlassCombinedLoss()(logits, targets).item() < 0.5


def test_combined_loss_exposes_both_components() -> None:
    torch.manual_seed(0)
    cross_entropy, dice = MulticlassCombinedLoss().components(
        torch.randn(2, 3, 32, 32), torch.randint(0, 3, (2, 32, 32))
    )

    assert cross_entropy.ndim == 0
    assert dice.ndim == 0


@pytest.mark.parametrize("class_weights", [None, torch.tensor([0.1, 2.0, 3.0])])
def test_combined_loss_propagates_gradients(class_weights: torch.Tensor | None) -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    MulticlassCombinedLoss(class_weights=class_weights)(logits, torch.randint(0, 3, (2, 32, 32))).backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0
