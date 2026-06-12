from __future__ import annotations

import numpy as np

from prometheus.data.transforms import RandomBrightnessContrast, RandomChannelJitter, RandomGaussianNoise


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
