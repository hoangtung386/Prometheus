from __future__ import annotations

import math

import torch

from prometheus.metrics import SegmentationEvaluator
from prometheus.training import dice_score, warmup_cosine_lr


def _logits_from_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = torch.full((1, num_classes, *mask.shape), -10.0)
    for cls_idx in range(num_classes):
        logits[0, cls_idx][mask == cls_idx] = 10.0
    return logits


def test_evaluator_ignores_absent_class_in_mean() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    logits = _logits_from_mask(target[0], num_classes=3)
    evaluator = SegmentationEvaluator(num_classes=3)
    evaluator.update(logits, target)
    metrics = evaluator.compute()
    assert math.isnan(metrics["dice"][1].item())
    assert evaluator.log_dict("x")["x/Dice/mean_present_fg"] == 0.0


def test_evaluator_false_positive_absent_class_is_zero() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    pred = target.clone()
    pred[:, :2, :2] = 1
    evaluator = SegmentationEvaluator(num_classes=3)
    evaluator.update(_logits_from_mask(pred[0], 3), target)
    metrics = evaluator.compute()
    assert metrics["dice"][1].item() == 0.0


def test_evaluator_missed_present_class_is_zero() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :2, :2] = 1
    pred = torch.zeros_like(target)
    evaluator = SegmentationEvaluator(num_classes=3)
    evaluator.update(_logits_from_mask(pred[0], 3), target)
    metrics = evaluator.compute()
    assert metrics["dice"][1].item() == 0.0


def test_dice_score_present_foreground_only() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :2, :2] = 1
    logits = _logits_from_mask(target[0], num_classes=3)
    score = dice_score(logits, target)
    assert score.item() == 1.0


def test_warmup_cosine_lr_respects_min_ratio() -> None:
    factor = warmup_cosine_lr(epoch=10, warmup_epochs=0, total_epochs=10, min_lr_ratio=0.1)
    assert abs(factor - 0.1) < 1e-6
