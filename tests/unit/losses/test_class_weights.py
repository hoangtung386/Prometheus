from __future__ import annotations

import pytest
import torch

from prometheus.domain import ImageMeta, MultitaskBatch, NucleiTarget, TissueTarget
from prometheus.losses import BACKGROUND_TISSUE_WEIGHT, class_weights_from_counts, compute_class_weights


def test_rare_classes_get_more_weight() -> None:
    weights = class_weights_from_counts(torch.tensor([900.0, 90.0, 10.0, 0.0]))
    assert weights[0] < weights[1] < weights[2]
    assert weights[:3].mean().item() == pytest.approx(1.0, abs=1e-5)
    assert weights[3].item() == 1.0, "an absent class must stay neutral"


def test_all_absent_is_ones() -> None:
    assert torch.allclose(class_weights_from_counts(torch.zeros(5)), torch.ones(5))


def test_square_root_power_compresses_the_dynamic_range() -> None:
    counts = torch.tensor([1_000_000.0, 1_000.0])
    plain = class_weights_from_counts(counts, power=1.0)
    root = class_weights_from_counts(counts, power=0.5)
    assert (plain[1] / plain[0]) > (root[1] / root[0]) > 1.0


def test_reserved_prefix_pins_background_and_excludes_it_from_normalization() -> None:
    # Background is the rarest tissue label but is not scored by the challenge, so plain
    # inverse frequency would hand it the largest weight. It must be pinned instead.
    counts = torch.tensor([50.0, 900_000.0, 600_000.0, 90_000.0, 20_000.0, 25_000.0])
    weights = class_weights_from_counts(counts, reserved_prefix=1)
    assert weights[0] == BACKGROUND_TISSUE_WEIGHT
    assert weights[0] < weights[1:].min(), "background must never outweigh a scored class"
    assert weights[1:].mean().item() == pytest.approx(1.0, abs=1e-5)
    assert weights[4] > weights[1], "necrosis is rarer than tumour and must weigh more"


def test_rejects_invalid_arguments() -> None:
    counts = torch.ones(3)
    with pytest.raises(ValueError, match="power"):
        class_weights_from_counts(counts, power=0.0)
    with pytest.raises(ValueError, match="reserved_prefix"):
        class_weights_from_counts(counts, reserved_prefix=3)


def _batch() -> MultitaskBatch:
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    mask[0, 0, :] = 1
    meta = ImageMeta("sample", (4, 4), (4, 4), (4, 4), (1.0, 1.0), (0, 0))
    return MultitaskBatch(
        images=torch.zeros(1, 3, 4, 4),
        tissue=TissueTarget(mask),
        nuclei=[NucleiTarget(centroids=torch.zeros(2, 2), labels=torch.tensor([0, 2]), boxes=torch.zeros(2, 4))],
        metadata=[meta],
    )


def test_compute_class_weights_shapes_and_background_handling() -> None:
    tissue_weights, nuclei_weights = compute_class_weights(
        [_batch(), _batch()], num_tissue_classes=6, num_nucleus_types=10
    )
    assert tissue_weights.shape == (6,)
    assert nuclei_weights.shape == (10,)
    assert torch.isfinite(tissue_weights).all()
    assert torch.isfinite(nuclei_weights).all()
    assert tissue_weights[0] == BACKGROUND_TISSUE_WEIGHT
