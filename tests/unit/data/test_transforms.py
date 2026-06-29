from __future__ import annotations

import numpy as np

from prometheus.data.transforms import RandomBrightnessContrast, RandomChannelJitter, RandomGaussianNoise
from prometheus.data.transforms.detection import horizontal_flip_points, vertical_flip_points


def test_color_transforms_keep_image_in_unit_range() -> None:
    image = np.full((3, 16, 16), 0.5, dtype=np.float32)
    for transform in [
        RandomBrightnessContrast(brightness=0.2, contrast=0.5),
        RandomChannelJitter(scale=0.5, shift=0.2, p=1.0),
        RandomGaussianNoise(std=0.5),
    ]:
        result = transform(image=image)
        image = result["image"]
        assert image.min() >= 0.0
        assert image.max() <= 1.0


def test_detection_point_flips_preserve_pixel_coordinate_convention() -> None:
    points = np.array([[1.0, 2.0], [8.0, 7.0]], dtype=np.float32)
    assert np.allclose(horizontal_flip_points(points, width=10), [[8, 2], [1, 7]])
    assert np.allclose(vertical_flip_points(points, height=10), [[1, 7], [8, 2]])
