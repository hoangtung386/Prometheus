"""Deterministic one-to-one centroid matching for nuclei detections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..domain import Detection, NucleusInstance


@dataclass(frozen=True)
class Match:
    prediction_index: int
    target_index: int
    distance_px: float


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[Match, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]


def _label_value(label) -> str:
    return label.value if hasattr(label, "value") else str(label)


def match_detections(
    predictions: Sequence[Detection],
    targets: Sequence[NucleusInstance | Detection],
    radius_px: float = 15.0,
    require_class_match: bool = True,
) -> MatchResult:
    """Match exactly like the official PUMA evaluator.

    Ground-truth instances are processed in input order. For each target, the
    eligible unused predictions have the same class and distance strictly less
    than ``radius_px``. The candidate with highest confidence and then nearest
    distance is selected.
    """

    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    unused_predictions = set(range(len(predictions)))
    matches = []
    unmatched_targets = []
    for target_index, target in enumerate(targets):
        candidates = []
        for prediction_index in unused_predictions:
            prediction = predictions[prediction_index]
            if require_class_match and _label_value(prediction.label) != _label_value(target.label):
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(prediction.centroid) - np.asarray(target.centroid),
                )
            )
            if distance < radius_px:
                candidates.append((-prediction.confidence, distance, prediction_index))
        if not candidates:
            unmatched_targets.append(target_index)
            continue
        _, distance, prediction_index = min(candidates)
        unused_predictions.remove(prediction_index)
        matches.append(Match(prediction_index, target_index, distance))
    return MatchResult(
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(sorted(unused_predictions)),
        unmatched_target_indices=tuple(unmatched_targets),
    )
