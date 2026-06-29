"""PUMA-style nuclei detection metrics aggregated from TP/FP/FN counts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ..domain import Detection, NucleusInstance
from .matching import match_detections


def _label_value(label) -> str:
    return label.value if hasattr(label, "value") else str(label)


def nuclei_detection_metrics(
    predictions: Sequence[Sequence[Detection]],
    targets: Sequence[Sequence[NucleusInstance]],
    radius_px: float = 15.0,
) -> dict[str, object]:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must contain the same number of samples")
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    image_macro_f1 = []
    all_classes = sorted({_label_value(item.label) for collection in [*predictions, *targets] for item in collection})
    for sample_predictions, sample_targets in zip(predictions, targets, strict=True):
        result = match_detections(sample_predictions, sample_targets, radius_px)
        image_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for match in result.matches:
            label = _label_value(sample_targets[match.target_index].label)
            counts[label]["tp"] += 1
            image_counts[label]["tp"] += 1
        for index in result.unmatched_prediction_indices:
            label = _label_value(sample_predictions[index].label)
            counts[label]["fp"] += 1
            image_counts[label]["fp"] += 1
        for index in result.unmatched_target_indices:
            label = _label_value(sample_targets[index].label)
            counts[label]["fn"] += 1
            image_counts[label]["fn"] += 1
        scores = [_f1(**image_counts[label]) for label in all_classes]
        image_macro_f1.append(sum(scores) / len(scores) if scores else 0.0)

    per_class = {}
    for label in all_classes:
        class_counts = counts[label]
        tp, fp, fn = class_counts["tp"], class_counts["fp"], class_counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            **class_counts,
            "precision": precision,
            "recall": recall,
            "f1": _f1(tp, fp, fn),
        }
    macro_f1 = sum(value["f1"] for value in per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "per_class": per_class,
        "macro_f1_summed": macro_f1,
        "macro_f1_per_image": (sum(image_macro_f1) / len(image_macro_f1) if image_macro_f1 else 0.0),
    }


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0
