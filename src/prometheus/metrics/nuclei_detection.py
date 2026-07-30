"""Nuclei detection metrics aggregated from pooled TP/FP/FN counts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..domain import Detection, NucleusClass, NucleusInstance
from .matching import match_detections

__all__ = ["ClassCounts", "NucleiDetectionMetrics", "nuclei_detection_metrics"]


@dataclass(frozen=True)
class ClassCounts:
    """Pooled counts and derived scores for one nucleus class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        """Share of predictions that hit a target."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        """Share of targets that were hit."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall; 0.0 when the class is absent from both."""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0


@dataclass(frozen=True)
class NucleiDetectionMetrics:
    """Nuclei detection outcome over a whole prediction set.

    Attributes:
        per_class: Pooled counts per class name.
        macro_f1_summed: Mean F1 over classes, computed from set-wide pooled counts. This
            is the challenge-shaped metric.
        macro_f1_per_image: Mean over images of each image's macro F1. Reported for
            diagnosis only; it is dominated by whichever images contain few nuclei.
    """

    per_class: dict[str, ClassCounts] = field(default_factory=dict)
    macro_f1_summed: float = 0.0
    macro_f1_per_image: float = 0.0


def _label_value(label: NucleusClass | str) -> str:
    return label.value if isinstance(label, NucleusClass) else label


def _macro_f1(counts: dict[str, ClassCounts], class_names: Sequence[str]) -> float:
    if not class_names:
        return 0.0
    return sum(counts.get(name, ClassCounts()).f1 for name in class_names) / len(class_names)


def nuclei_detection_metrics(
    predictions: Sequence[Sequence[Detection]],
    targets: Sequence[Sequence[NucleusInstance | Detection]],
    radius_px: float = 15.0,
) -> NucleiDetectionMetrics:
    """Match detections to ground truth per image and pool the counts.

    Classes are taken from the union of predicted and annotated labels, so a class the model
    only hallucinates still contributes its false positives to the macro average.
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Got {len(predictions)} prediction lists for {len(targets)} target lists")

    class_names = sorted({_label_value(item.label) for collection in (*predictions, *targets) for item in collection})
    pooled: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_image_scores: list[float] = []

    for sample_predictions, sample_targets in zip(predictions, targets, strict=True):
        result = match_detections(sample_predictions, sample_targets, radius_px)
        image: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for match in result.matches:
            label = _label_value(sample_targets[match.target_index].label)
            pooled[label][0] += 1
            image[label][0] += 1
        for index in result.unmatched_prediction_indices:
            label = _label_value(sample_predictions[index].label)
            pooled[label][1] += 1
            image[label][1] += 1
        for index in result.unmatched_target_indices:
            label = _label_value(sample_targets[index].label)
            pooled[label][2] += 1
            image[label][2] += 1
        image_counts = {name: ClassCounts(*values) for name, values in image.items()}
        per_image_scores.append(_macro_f1(image_counts, class_names))

    per_class = {name: ClassCounts(*pooled[name]) for name in class_names}
    return NucleiDetectionMetrics(
        per_class=per_class,
        macro_f1_summed=_macro_f1(per_class, class_names),
        macro_f1_per_image=(sum(per_image_scores) / len(per_image_scores) if per_image_scores else 0.0),
    )
