"""Synchronized augmentation of images, tissue masks and nuclei geometry.

Every geometric transform must update the image, the tissue mask, the nuclei centroids and
the nuclei boxes together, which is why this is hand-written rather than delegated: an
image-only augmentation library silently desynchronises the instance geometry.

Known gaps, both material on a 205 image dataset — see
``docs/phan-tich-tissue-va-ke-hoach.md`` section 3.8:

* no random scaling or free rotation (only the eight dihedral views);
* stain jitter is per-channel gain/bias in RGB, not a stain-matrix perturbation.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "MultitaskCompose",
    "NormalizeMultitask",
    "RandomBrightnessContrastMultitask",
    "RandomGammaMultitask",
    "RandomGaussianNoiseMultitask",
    "RandomHorizontalFlipMultitask",
    "RandomRotate90Multitask",
    "RandomStainJitterMultitask",
    "RandomVerticalFlipMultitask",
    "TransformSample",
    "multitask_train_transform",
    "multitask_validation_transform",
]


@dataclass(frozen=True)
class TransformSample:
    """One sample in transform space: ``image`` is CHW, geometry is pixel-space ``(x, y)``.

    ``valid_mask`` marks the letterboxed content area so padding is excluded from
    normalization instead of being normalized into a non-zero constant.
    """

    image: np.ndarray
    tissue_mask: np.ndarray
    centroids: np.ndarray
    boxes: np.ndarray
    valid_mask: np.ndarray | None = None


class MultitaskCompose:
    """Apply transforms in order."""

    def __init__(self, transforms: Sequence[Callable[[TransformSample], TransformSample]]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, sample: TransformSample) -> TransformSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class _GeometricMultitask:
    """Base for transforms that move pixels: image, mask, valid mask and geometry together.

    Subclasses implement :meth:`_transform_array` for the dense arrays and
    :meth:`_transform_points` for ``(N, 2)`` pixel coordinates. Boxes are derived from their
    transformed corners, which keeps them axis-aligned under any dihedral view.
    """

    probability = 1.0

    def _draw(self) -> bool:
        return random.random() < self.probability

    def _transform_array(self, array: np.ndarray, spatial_axes: tuple[int, int]) -> np.ndarray:
        raise NotImplementedError

    def _transform_points(self, points: np.ndarray, height: int, width: int) -> np.ndarray:
        raise NotImplementedError

    def _transform_boxes(self, boxes: np.ndarray, height: int, width: int) -> np.ndarray:
        if not boxes.size:
            return boxes.copy()
        corners = np.stack(
            (boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]),
            axis=1,
        )
        moved = self._transform_points(corners.reshape(-1, 2), height, width).reshape(-1, 4, 2)
        return np.concatenate((moved.min(axis=1), moved.max(axis=1)), axis=1)

    def __call__(self, sample: TransformSample) -> TransformSample:
        if not self._draw():
            return sample
        height, width = sample.image.shape[1:]
        return TransformSample(
            image=self._transform_array(sample.image, (1, 2)),
            tissue_mask=self._transform_array(sample.tissue_mask, (0, 1)),
            centroids=self._transform_points(sample.centroids, height, width),
            boxes=self._transform_boxes(sample.boxes, height, width),
            valid_mask=(None if sample.valid_mask is None else self._transform_array(sample.valid_mask, (0, 1))),
        )


class RandomHorizontalFlipMultitask(_GeometricMultitask):
    """Mirror along the vertical axis."""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def _transform_array(self, array: np.ndarray, spatial_axes: tuple[int, int]) -> np.ndarray:
        return np.flip(array, axis=spatial_axes[1]).copy()

    def _transform_points(self, points: np.ndarray, height: int, width: int) -> np.ndarray:  # noqa: ARG002
        moved = points.copy()
        moved[:, 0] = width - 1 - moved[:, 0]
        return moved


class RandomVerticalFlipMultitask(_GeometricMultitask):
    """Mirror along the horizontal axis."""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def _transform_array(self, array: np.ndarray, spatial_axes: tuple[int, int]) -> np.ndarray:
        return np.flip(array, axis=spatial_axes[0]).copy()

    def _transform_points(self, points: np.ndarray, height: int, width: int) -> np.ndarray:  # noqa: ARG002
        moved = points.copy()
        moved[:, 1] = height - 1 - moved[:, 1]
        return moved


class RandomRotate90Multitask(_GeometricMultitask):
    """Rotate by a uniformly drawn multiple of 90 degrees.

    ``np.rot90`` by ``k`` over ``(H, W)`` moves the pixel at ``(x, y)`` to
    ``(y, W-1-x)``, applied ``k`` times. The point maps below spell that out per ``k``
    rather than composing it, and stay correct for non-square inputs because the output
    axis lengths swap on odd ``k``.
    """

    def __init__(self) -> None:
        self._rotations = 0

    def _draw(self) -> bool:
        self._rotations = random.choice((0, 1, 2, 3))
        return self._rotations != 0

    def _transform_array(self, array: np.ndarray, spatial_axes: tuple[int, int]) -> np.ndarray:
        return np.rot90(array, self._rotations, axes=spatial_axes).copy()

    def _transform_points(self, points: np.ndarray, height: int, width: int) -> np.ndarray:
        moved = points.copy()
        x_coordinate, y_coordinate = points[:, 0].copy(), points[:, 1].copy()
        if self._rotations == 1:
            moved[:, 0], moved[:, 1] = y_coordinate, width - 1 - x_coordinate
        elif self._rotations == 2:
            moved[:, 0], moved[:, 1] = width - 1 - x_coordinate, height - 1 - y_coordinate
        else:
            moved[:, 0], moved[:, 1] = height - 1 - y_coordinate, x_coordinate
        return moved


class _PhotometricMultitask:
    """Base for image-only transforms: geometry (masks/boxes/points) is untouched.

    Operates on the raw image (CHW float in ``[0, 1]``) and must run before
    :class:`NormalizeMultitask`. Output is clipped back to ``[0, 1]``.
    """

    def _apply(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def __call__(self, sample: TransformSample) -> TransformSample:
        return replace(sample, image=np.clip(self._apply(sample.image.astype(np.float32)), 0.0, 1.0))


class RandomStainJitterMultitask(_PhotometricMultitask):
    """Per-channel gain/bias — a cheap stain-variation proxy for H&E tiles."""

    def __init__(self, scale: float = 0.08, shift: float = 0.03, probability: float = 0.8) -> None:
        self.scale = scale
        self.shift = shift
        self.probability = probability

    def _apply(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return image
        shape = (image.shape[0], 1, 1)
        gains = np.random.uniform(1 - self.scale, 1 + self.scale, size=shape).astype(np.float32)
        shifts = np.random.uniform(-self.shift, self.shift, size=shape).astype(np.float32)
        return image * gains + shifts


class RandomBrightnessContrastMultitask(_PhotometricMultitask):
    """Linear intensity rescale."""

    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, probability: float = 0.5) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.probability = probability

    def _apply(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return image
        brightness = random.uniform(-self.brightness, self.brightness)
        contrast = 1.0 + random.uniform(-self.contrast, self.contrast)
        return image * contrast + brightness


class RandomGammaMultitask(_PhotometricMultitask):
    """Non-linear intensity rescale."""

    def __init__(self, gamma_range: tuple[float, float] = (0.8, 1.25), probability: float = 0.5) -> None:
        self.gamma_range = gamma_range
        self.probability = probability

    def _apply(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return image
        gamma = random.uniform(*self.gamma_range)
        return np.clip(image, 0.0, 1.0) ** gamma


class RandomGaussianNoiseMultitask(_PhotometricMultitask):
    """Additive sensor-noise proxy."""

    def __init__(self, std: float = 0.01, probability: float = 0.5) -> None:
        self.std = std
        self.probability = probability

    def _apply(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return image
        noise = np.random.randn(*image.shape).astype(np.float32) * self.std
        return image + noise


class NormalizeMultitask:
    """Normalize with the ImageNet statistics the pretrained ConvNeXt-V2 encoder expects.

    Only pixels inside ``valid_mask`` are normalized; letterbox padding is left at exactly
    zero so it maps to the encoder's mean colour instead of to a bright constant.
    """

    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    def __call__(self, sample: TransformSample) -> TransformSample:
        image = np.zeros_like(sample.image)
        valid_mask = sample.valid_mask
        if valid_mask is None:
            valid_mask = np.ones(sample.image.shape[1:], dtype=bool)
        for channel_index in range(image.shape[0]):
            source = sample.image[channel_index]
            image[channel_index, valid_mask] = (source[valid_mask] - self.mean[channel_index]) / self.std[channel_index]
        return replace(sample, image=image)


def multitask_train_transform() -> MultitaskCompose:
    """Training augmentation: dihedral geometry, then photometric, then normalization.

    Order matters. The photometric block operates on the raw ``[0, 1]`` image and must run
    before normalization; colour augmentation is the highest-value augment available on a
    dataset this small.
    """
    return MultitaskCompose(
        [
            RandomHorizontalFlipMultitask(),
            RandomVerticalFlipMultitask(),
            RandomRotate90Multitask(),
            RandomStainJitterMultitask(),
            RandomBrightnessContrastMultitask(),
            RandomGammaMultitask(),
            RandomGaussianNoiseMultitask(),
            NormalizeMultitask(),
        ]
    )


def multitask_validation_transform() -> MultitaskCompose:
    """Validation preprocessing: normalization only, no augmentation."""
    return MultitaskCompose([NormalizeMultitask()])
