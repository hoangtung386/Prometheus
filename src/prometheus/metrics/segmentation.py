"""Tissue segmentation metrics, including the official PUMA micro Dice.

The challenge ranks tissue segmentation with a **micro Dice**: all predictions are
concatenated into a single volume, per-class Dice is computed on those pooled counts, and
the five scored classes are averaged. ``tissue_white_background`` is excluded.

Two mistakes in a naive implementation both inflate the reported score, and both were
present in earlier revisions of this project:

* returning ``1.0`` for a class that is absent from *both* prediction and ground truth —
  the official metric awards nothing for a class nobody mentioned;
* dropping such a class from the average (``nanmean``) — which silently hides a class the
  model has never learned. ``necrosis`` sat at exactly ``0.0`` for a full training run
  without a single log line reporting it.

:class:`SegmentationEvaluator` accumulates pooled counts, so
``metrics["tissue/micro_dice"]`` is the leaderboard metric computed on the validation
fold, and every class is always reported.
"""

from __future__ import annotations

import torch

from ..domain import TISSUE_CLASS_NAMES, scored_tissue_classes

__all__ = ["SegmentationEvaluator", "official_micro_dice"]

_EPS = 1e-8


def _dice(true_positive: float, false_positive: float, false_negative: float) -> float:
    """Dice from pooled counts; ``0.0`` when the class appears in neither input."""
    denominator = 2.0 * true_positive + false_positive + false_negative
    return (2.0 * true_positive / denominator) if denominator > 0 else 0.0


def official_micro_dice(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, float]:
    """Compute the official PUMA tissue micro Dice over a whole prediction set.

    Args:
        predictions: Per-image ``(H, W)`` masks of training class indices.
        targets: Per-image ``(H, W)`` ground-truth masks of training class indices.

    Returns:
        Per-class Dice keyed by ``tissue_<name>`` for the five scored classes, plus
        ``average_micro_dice``. Absent classes score ``0.0`` and are still averaged, which
        is what the leaderboard does.
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Got {len(predictions)} predictions for {len(targets)} targets")

    evaluator = SegmentationEvaluator()
    for prediction, target in zip(predictions, targets, strict=True):
        evaluator.update_from_labels(prediction, target)
    scores = evaluator.micro_dice()
    scores["average_micro_dice"] = evaluator.average_micro_dice()
    return scores


class SegmentationEvaluator:
    """Accumulate pooled per-class TP/FP/FN/TN counts across batches.

    Pooling across the whole set (rather than averaging per-image scores) is what makes
    the result a *micro* metric and therefore comparable with the leaderboard.
    """

    def __init__(
        self,
        num_classes: int = len(TISSUE_CLASS_NAMES),
        class_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        names = tuple(class_names) if class_names is not None else TISSUE_CLASS_NAMES[:num_classes]
        if len(names) != num_classes:
            raise ValueError(f"Got {len(names)} class names for {num_classes} classes")
        self.num_classes = num_classes
        self.class_names = names
        self.reset()

    def reset(self) -> None:
        zeros = torch.zeros(self.num_classes, dtype=torch.float64)
        self.true_positive = zeros.clone()
        self.false_positive = zeros.clone()
        self.false_negative = zeros.clone()
        self.true_negative = zeros.clone()
        self.target_count = zeros.clone()
        self.prediction_count = zeros.clone()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch of ``(B, C, H, W)`` logits against ``(B, H, W)`` labels."""
        self.update_from_labels(logits.argmax(dim=1), targets)

    def update_from_labels(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate already-decoded class-index predictions against class-index labels."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Prediction shape {tuple(predictions.shape)} != target shape {tuple(targets.shape)}")
        predictions = predictions.detach().reshape(-1)
        targets = targets.detach().reshape(-1).to(predictions.device)
        for class_index in range(self.num_classes):
            predicted = predictions == class_index
            actual = targets == class_index
            self.true_positive[class_index] += float((predicted & actual).sum())
            self.false_positive[class_index] += float((predicted & ~actual).sum())
            self.false_negative[class_index] += float((~predicted & actual).sum())
            self.true_negative[class_index] += float((~predicted & ~actual).sum())
            self.target_count[class_index] += float(actual.sum())
            self.prediction_count[class_index] += float(predicted.sum())

    def micro_dice(self) -> dict[str, float]:
        """Per-class micro Dice for the five scored tissue classes, keyed ``tissue_<name>``."""
        return {
            f"tissue_{name}": _dice(
                float(self.true_positive[index]),
                float(self.false_positive[index]),
                float(self.false_negative[index]),
            )
            for name, index in scored_tissue_classes()
            if index < self.num_classes
        }

    def average_micro_dice(self) -> float:
        """The official PUMA tissue score: mean micro Dice over the scored classes."""
        scores = self.micro_dice()
        return sum(scores.values()) / len(scores) if scores else 0.0

    def compute(self) -> dict[str, torch.Tensor]:
        """Return per-class Dice, IoU, sensitivity, specificity, precision and support."""
        dice_denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        iou_denominator = self.true_positive + self.false_positive + self.false_negative
        recall_denominator = self.true_positive + self.false_negative
        precision_denominator = self.true_positive + self.false_positive
        return {
            "dice": 2 * self.true_positive / dice_denominator.clamp_min(_EPS),
            "iou": self.true_positive / iou_denominator.clamp_min(_EPS),
            "sensitivity": self.true_positive / recall_denominator.clamp_min(_EPS),
            "precision": self.true_positive / precision_denominator.clamp_min(_EPS),
            "specificity": self.true_negative / (self.true_negative + self.false_positive).clamp_min(_EPS),
            "target_count": self.target_count.clone(),
            "prediction_count": self.prediction_count.clone(),
        }

    def log_dict(self, prefix: str = "tissue") -> dict[str, float]:
        """Flatten every per-class metric plus the official score into a log-friendly dict.

        ``<prefix>/micro_dice`` is the leaderboard metric. The per-class
        ``<prefix>/dice/<name>`` entries exist so a collapsed class is visible in the
        training log the epoch it happens.
        """
        metrics = self.compute()
        log: dict[str, float] = {}
        for index, name in enumerate(self.class_names):
            log[f"{prefix}/dice/{name}"] = float(metrics["dice"][index])
            log[f"{prefix}/iou/{name}"] = float(metrics["iou"][index])
            log[f"{prefix}/sensitivity/{name}"] = float(metrics["sensitivity"][index])
            log[f"{prefix}/precision/{name}"] = float(metrics["precision"][index])
            log[f"{prefix}/support/target/{name}"] = float(metrics["target_count"][index])
            log[f"{prefix}/support/prediction/{name}"] = float(metrics["prediction_count"][index])
        for name, score in self.micro_dice().items():
            log[f"{prefix}/micro_dice/{name.removeprefix('tissue_')}"] = score
        log[f"{prefix}/micro_dice"] = self.average_micro_dice()
        return log
