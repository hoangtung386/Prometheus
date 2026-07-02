from __future__ import annotations

import torch

from prometheus.domain import ImageMeta, MultitaskBatch, NucleiTarget, TissueTarget
from prometheus.losses import compute_class_weights, inverse_frequency_weights


def test_inverse_frequency_gives_rare_classes_more_weight() -> None:
    weights = inverse_frequency_weights(torch.tensor([900.0, 90.0, 10.0, 0.0]))
    assert weights[0] < weights[1] < weights[2]  # rarer -> heavier
    assert abs(weights[:3].mean().item() - 1.0) < 1e-5  # mean 1 over present classes
    assert weights[3].item() == 1.0  # absent class is neutral


def test_inverse_frequency_all_absent_is_ones() -> None:
    assert torch.allclose(inverse_frequency_weights(torch.zeros(5)), torch.ones(5))


def _batch() -> MultitaskBatch:
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    mask[0, 0, :] = 1  # a few class-1 pixels among class-0
    meta = ImageMeta("sample", (4, 4), (4, 4), (4, 4), (1.0, 1.0), (0, 0))
    return MultitaskBatch(
        images=torch.zeros(1, 3, 4, 4),
        tissue=TissueTarget(mask),
        nuclei=[NucleiTarget(centroids=torch.zeros(2, 2), labels=torch.tensor([0, 2]), boxes=torch.zeros(2, 4))],
        metadata=[meta],
    )


def test_compute_class_weights_shapes_and_finiteness() -> None:
    tissue_weights, nuclei_weights = compute_class_weights(
        [_batch(), _batch()], num_tissue_classes=6, num_nucleus_types=10
    )
    assert tissue_weights.shape == (6,)
    assert nuclei_weights.shape == (10,)
    assert torch.isfinite(tissue_weights).all() and torch.isfinite(nuclei_weights).all()
