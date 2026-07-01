"""Synchronized transforms for images, tissue masks, and nuclei geometry."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class TransformSample:
    image: np.ndarray
    tissue_mask: np.ndarray
    centroids: np.ndarray
    boxes: np.ndarray
    valid_mask: np.ndarray | None = None


class MultitaskCompose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: TransformSample) -> TransformSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomHorizontalFlipMultitask:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, sample: TransformSample) -> TransformSample:
        if random.random() >= self.probability:
            return sample
        width = sample.image.shape[2]
        centroids = sample.centroids.copy()
        centroids[:, 0] = width - 1 - centroids[:, 0]
        boxes = sample.boxes.copy()
        if boxes.size:
            old_min, old_max = boxes[:, 0].copy(), boxes[:, 2].copy()
            boxes[:, 0], boxes[:, 2] = width - 1 - old_max, width - 1 - old_min
        return TransformSample(
            image=np.flip(sample.image, axis=2).copy(),
            tissue_mask=np.flip(sample.tissue_mask, axis=1).copy(),
            centroids=centroids,
            boxes=boxes,
            valid_mask=(
                np.flip(sample.valid_mask, axis=1).copy()
                if sample.valid_mask is not None
                else None
            ),
        )


class RandomVerticalFlipMultitask:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, sample: TransformSample) -> TransformSample:
        if random.random() >= self.probability:
            return sample
        height = sample.image.shape[1]
        centroids = sample.centroids.copy()
        centroids[:, 1] = height - 1 - centroids[:, 1]
        boxes = sample.boxes.copy()
        if boxes.size:
            old_min, old_max = boxes[:, 1].copy(), boxes[:, 3].copy()
            boxes[:, 1], boxes[:, 3] = height - 1 - old_max, height - 1 - old_min
        return TransformSample(
            image=np.flip(sample.image, axis=1).copy(),
            tissue_mask=np.flip(sample.tissue_mask, axis=0).copy(),
            centroids=centroids,
            boxes=boxes,
            valid_mask=(
                np.flip(sample.valid_mask, axis=0).copy()
                if sample.valid_mask is not None
                else None
            ),
        )


class RandomRotate90Multitask:
    def __call__(self, sample: TransformSample) -> TransformSample:
        rotations = random.choice((0, 1, 2, 3))
        if rotations == 0:
            return sample
        height, width = sample.image.shape[1:]

        def rotate_points(points: np.ndarray) -> np.ndarray:
            points = points.copy()
            x_coord, y_coord = points[:, 0].copy(), points[:, 1].copy()
            if rotations == 1:
                points[:, 0], points[:, 1] = y_coord, width - 1 - x_coord
            elif rotations == 2:
                points[:, 0], points[:, 1] = width - 1 - x_coord, height - 1 - y_coord
            else:
                points[:, 0], points[:, 1] = height - 1 - y_coord, x_coord
            return points

        centroids = rotate_points(sample.centroids)
        if sample.boxes.size:
            boxes = sample.boxes
            corners = np.stack(
                (
                    boxes[:, [0, 1]],
                    boxes[:, [2, 1]],
                    boxes[:, [2, 3]],
                    boxes[:, [0, 3]],
                ),
                axis=1,
            )
            corners = rotate_points(corners.reshape(-1, 2)).reshape(-1, 4, 2)
            boxes = np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)
        else:
            boxes = sample.boxes.copy()
        return TransformSample(
            image=np.rot90(sample.image, rotations, axes=(1, 2)).copy(),
            tissue_mask=np.rot90(sample.tissue_mask, rotations, axes=(0, 1)).copy(),
            centroids=centroids,
            boxes=boxes,
            valid_mask=(
                np.rot90(sample.valid_mask, rotations, axes=(0, 1)).copy()
                if sample.valid_mask is not None
                else None
            ),
        )


class NormalizeMultitask:
    def __call__(self, sample: TransformSample) -> TransformSample:
        image = np.zeros_like(sample.image)
        valid_mask = sample.valid_mask
        if valid_mask is None:
            valid_mask = np.ones(sample.image.shape[1:], dtype=bool)
        for channel_index in range(image.shape[0]):
            source = sample.image[channel_index]
            valid_values = source[valid_mask]
            if valid_values.size == 0:
                continue
            low, high = np.percentile(valid_values, (2, 98))
            valid_values = np.clip(valid_values, low, high)
            normalized = (valid_values - valid_values.mean()) / (valid_values.std() + 1e-8)
            image[channel_index, valid_mask] = normalized
        return TransformSample(image, sample.tissue_mask, sample.centroids, sample.boxes, sample.valid_mask)


def multitask_train_transform() -> MultitaskCompose:
    return MultitaskCompose(
        [
            RandomHorizontalFlipMultitask(),
            RandomVerticalFlipMultitask(),
            RandomRotate90Multitask(),
            NormalizeMultitask(),
        ]
    )


def multitask_validation_transform() -> MultitaskCompose:
    return MultitaskCompose([NormalizeMultitask()])
