"""Nuclei matching and detection metrics, pinned against the official evaluator's rules."""

from __future__ import annotations

import numpy as np
import pytest

from prometheus.domain import Detection, NucleusClass, NucleusInstance
from prometheus.metrics import match_detections, nuclei_detection_metrics


def _target(x: float, y: float, label: NucleusClass = NucleusClass.TUMOR) -> NucleusInstance:
    polygon = np.array([[x - 1, y - 1], [x + 1, y - 1], [x, y + 1]])
    return NucleusInstance("x", label, polygon, (x, y), (x - 1, y - 1, x + 1, y + 1))


def test_radius_boundary_is_exclusive_like_official_evaluator() -> None:
    result = match_detections([Detection((15, 0), NucleusClass.TUMOR)], [_target(0, 0)], radius_px=15)

    assert len(result.matches) == 0, "the evaluator requires distance < radius, not <="


def test_highest_confidence_wins_before_nearest_distance() -> None:
    result = match_detections(
        [
            Detection((1, 0), NucleusClass.TUMOR, 0.5),
            Detection((10, 0), NucleusClass.TUMOR, 0.9),
        ],
        [_target(0, 0)],
    )

    assert result.matches[0].prediction_index == 1, "confidence outranks proximity"


def test_class_must_match() -> None:
    result = match_detections([Detection((0, 0), NucleusClass.STROMA)], [_target(0, 0)])

    assert len(result.matches) == 0
    assert result.unmatched_target_indices == (0,)


def test_duplicate_prediction_is_a_false_positive() -> None:
    metrics = nuclei_detection_metrics(
        [
            [
                Detection((0, 0), NucleusClass.TUMOR, 0.9),
                Detection((1, 0), NucleusClass.TUMOR, 0.8),
            ]
        ],
        [[_target(0, 0)]],
    )

    assert metrics.per_class["tumor"].tp == 1
    assert metrics.per_class["tumor"].fp == 1
    assert metrics.per_class["tumor"].f1 == pytest.approx(2 / 3)


def test_macro_average_includes_a_hallucinated_class() -> None:
    # A class the model invents but that never appears in the ground truth still drags the
    # macro average down through its false positives.
    metrics = nuclei_detection_metrics(
        [[Detection((0, 0), NucleusClass.TUMOR, 0.9), Detection((50, 50), NucleusClass.NEUTROPHIL, 0.9)]],
        [[_target(0, 0)]],
    )

    assert set(metrics.per_class) == {"tumor", "neutrophil"}
    assert metrics.per_class["neutrophil"].f1 == 0.0
    assert metrics.macro_f1_summed == pytest.approx(0.5)


def test_empty_input_is_zero_not_an_error() -> None:
    metrics = nuclei_detection_metrics([[]], [[]])

    assert metrics.per_class == {}
    assert metrics.macro_f1_summed == 0.0
    assert metrics.macro_f1_per_image == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="prediction lists"):
        nuclei_detection_metrics([[]], [[], []])
