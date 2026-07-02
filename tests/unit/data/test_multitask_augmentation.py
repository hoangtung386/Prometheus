from __future__ import annotations

import random

import numpy as np

from prometheus.data.transforms.multitask import (
    RandomBrightnessContrastMultitask,
    RandomGammaMultitask,
    RandomGaussianNoiseMultitask,
    RandomStainJitterMultitask,
    TransformSample,
    multitask_train_transform,
)

PHOTOMETRIC = [
    RandomStainJitterMultitask(probability=1.0),
    RandomBrightnessContrastMultitask(brightness=0.3, contrast=0.5, probability=1.0),
    RandomGammaMultitask(probability=1.0),
    RandomGaussianNoiseMultitask(std=0.5, probability=1.0),
]


def _sample() -> TransformSample:
    rng = np.random.default_rng(0)
    return TransformSample(
        image=rng.random((3, 32, 32), dtype=np.float32),
        tissue_mask=rng.integers(0, 6, (32, 32)).astype(np.int64),
        centroids=np.array([[4.0, 5.0], [20.0, 22.0]], dtype=np.float32),
        boxes=np.array([[2.0, 3.0, 6.0, 7.0], [18.0, 20.0, 22.0, 24.0]], dtype=np.float32),
        valid_mask=np.ones((32, 32), dtype=bool),
    )


def test_photometric_transforms_only_touch_pixels_and_stay_in_unit_range() -> None:
    base = _sample()
    for transform in PHOTOMETRIC:
        result = transform(base)
        assert np.array_equal(result.tissue_mask, base.tissue_mask)
        assert np.array_equal(result.centroids, base.centroids)
        assert np.array_equal(result.boxes, base.boxes)
        assert result.image.min() >= 0.0 and result.image.max() <= 1.0
        assert not np.allclose(result.image, base.image)  # active at probability=1


def test_train_transform_preserves_shapes_and_label_count() -> None:
    for seed in range(25):
        random.seed(seed)
        np.random.seed(seed)
        result = multitask_train_transform()(_sample())
        assert result.image.shape == (3, 32, 32)
        assert result.tissue_mask.shape == (32, 32)
        assert result.centroids.shape == (2, 2)
        assert result.boxes.shape == (2, 4)
        assert np.isfinite(result.image).all()
