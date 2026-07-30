"""The official PUMA tissue metric, including the two ways of inflating it.

See ``docs/phan-tich-tissue-va-ke-hoach.md`` sections 1.2 and 3.4.
"""

from __future__ import annotations

import pytest
import torch

from prometheus.domain import TISSUE_CLASS_NAMES, TISSUE_CLASS_TO_INDEX
from prometheus.metrics import SegmentationEvaluator, official_micro_dice

TUMOR = TISSUE_CLASS_TO_INDEX["tumor"]
NECROSIS = TISSUE_CLASS_TO_INDEX["necrosis"]
SCORED_CLASSES = 5


def _mask(value: int) -> torch.Tensor:
    return torch.full((4, 4), value, dtype=torch.long)


def test_absent_class_scores_zero_not_one() -> None:
    # A class in neither prediction nor ground truth earns nothing. Returning 1.0 here is
    # what turned a real 52.90 into an apparent 75.83.
    scores = official_micro_dice([_mask(TUMOR)], [_mask(TUMOR)])

    assert scores["tissue_tumor"] == pytest.approx(1.0)
    assert scores["tissue_necrosis"] == 0.0
    assert scores["average_micro_dice"] == pytest.approx(1.0 / SCORED_CLASSES)


def test_absent_class_is_averaged_rather_than_skipped() -> None:
    scores = official_micro_dice([_mask(TUMOR)], [_mask(TUMOR)])

    assert len(scores) == SCORED_CLASSES + 1, "every scored class must appear in the report"
    assert scores["average_micro_dice"] < 1.0, "a NaN-skipping average would report a perfect score"


def test_background_is_excluded_from_the_average() -> None:
    background = _mask(TISSUE_CLASS_TO_INDEX["background"])
    scores = official_micro_dice([background], [background])

    assert "tissue_background" not in scores
    assert scores["average_micro_dice"] == 0.0


def test_metric_is_micro_not_per_image_macro() -> None:
    # One large correct image and one small wrong one. A per-image macro average would give
    # 0.5; pooling the counts weights by area, as the leaderboard does.
    predictions = [torch.full((10, 10), TUMOR), torch.full((1, 1), NECROSIS)]
    targets = [torch.full((10, 10), TUMOR), torch.full((1, 1), TUMOR)]
    scores = official_micro_dice(predictions, targets)

    assert scores["tissue_tumor"] == pytest.approx(2 * 100 / (2 * 100 + 0 + 1))
    assert scores["tissue_necrosis"] == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="predictions"):
        official_micro_dice([_mask(TUMOR)], [_mask(TUMOR), _mask(TUMOR)])


def test_evaluator_reports_every_class_by_name() -> None:
    evaluator = SegmentationEvaluator(class_names=TISSUE_CLASS_NAMES)
    logits = torch.zeros(1, len(TISSUE_CLASS_NAMES), 4, 4)
    logits[:, TUMOR] = 10.0
    evaluator.update(logits, _mask(TUMOR).unsqueeze(0))
    log = evaluator.log_dict("tissue")

    # A collapsed rare class has to be visible in the log the epoch it happens.
    assert log["tissue/micro_dice/necrosis"] == 0.0
    assert log["tissue/micro_dice/tumor"] == pytest.approx(1.0)
    assert log["tissue/micro_dice"] == pytest.approx(1.0 / SCORED_CLASSES)
    assert log["tissue/support/target/tumor"] == 16.0


def test_evaluator_rejects_inconsistent_construction() -> None:
    with pytest.raises(ValueError, match="class names"):
        SegmentationEvaluator(num_classes=6, class_names=("a", "b"))
    with pytest.raises(ValueError, match="num_classes"):
        SegmentationEvaluator(num_classes=0)


def test_evaluator_reset_clears_counts() -> None:
    evaluator = SegmentationEvaluator()
    evaluator.update_from_labels(_mask(TUMOR), _mask(TUMOR))
    evaluator.reset()

    assert evaluator.average_micro_dice() == 0.0
