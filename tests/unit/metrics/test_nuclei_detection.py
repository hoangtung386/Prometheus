import numpy as np

from prometheus.domain import Detection, NucleusClass, NucleusInstance
from prometheus.metrics import match_detections, nuclei_detection_metrics


def _target(x: float, y: float, label=NucleusClass.TUMOR) -> NucleusInstance:
    polygon = np.array([[x - 1, y - 1], [x + 1, y - 1], [x, y + 1]])
    return NucleusInstance("x", label, polygon, (x, y), (x - 1, y - 1, x + 1, y + 1))


def test_radius_boundary_is_exclusive_like_official_evaluator() -> None:
    result = match_detections(
        [Detection((15, 0), NucleusClass.TUMOR)],
        [_target(0, 0)],
        radius_px=15,
    )
    assert len(result.matches) == 0


def test_highest_confidence_wins_before_nearest_distance() -> None:
    result = match_detections(
        [
            Detection((1, 0), NucleusClass.TUMOR, 0.5),
            Detection((10, 0), NucleusClass.TUMOR, 0.9),
        ],
        [_target(0, 0)],
    )
    assert result.matches[0].prediction_index == 1


def test_duplicate_prediction_is_false_positive() -> None:
    predictions = [
        [
            Detection((0, 0), NucleusClass.TUMOR, 0.9),
            Detection((1, 0), NucleusClass.TUMOR, 0.8),
        ]
    ]
    metrics = nuclei_detection_metrics(predictions, [[_target(0, 0)]])
    assert metrics["per_class"]["tumor"]["tp"] == 1
    assert metrics["per_class"]["tumor"]["fp"] == 1
