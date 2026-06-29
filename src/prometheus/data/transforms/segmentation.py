"""Synchronized transforms for channel-first images and segmentation masks."""

from __future__ import annotations

import random

import cv2
import numpy as np

from .common import (
    Compose,
    NormalizeTile,
    RandomBrightnessContrast,
    RandomChannelJitter,
    RandomGamma,
    RandomGaussianNoise,
)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, masks, **kwargs) -> dict:
        if random.random() < self.p:
            image = np.flip(image, axis=2).copy()
            masks = {key: np.flip(mask, axis=2).copy() for key, mask in masks.items()}
        return {"image": image, "masks": masks, **kwargs}


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, masks, **kwargs) -> dict:
        if random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            masks = {key: np.flip(mask, axis=1).copy() for key, mask in masks.items()}
        return {"image": image, "masks": masks, **kwargs}


class RandomRotate90:
    def __call__(self, image, masks, **kwargs) -> dict:
        rotations = random.choice([0, 1, 2, 3])
        if rotations:
            image = np.rot90(image, rotations, axes=(1, 2)).copy()
            masks = {key: np.rot90(mask, rotations, axes=(1, 2)).copy() for key, mask in masks.items()}
        return {"image": image, "masks": masks, **kwargs}


class ElasticDeformation:
    def __init__(self, alpha: float = 30, sigma: float = 4, p: float = 0.3) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, masks, **kwargs) -> dict:
        if random.random() > self.p:
            return {"image": image, "masks": masks, **kwargs}
        height, width = image.shape[1:]
        dx = (
            cv2.GaussianBlur(
                np.random.rand(height, width) * 2 - 1,
                (0, 0),
                self.sigma,
            )
            * self.alpha
        )
        dy = (
            cv2.GaussianBlur(
                np.random.rand(height, width) * 2 - 1,
                (0, 0),
                self.sigma,
            )
            * self.alpha
        )
        x_map, y_map = np.meshgrid(np.arange(width), np.arange(height))
        x_map = (x_map + dx).astype(np.float32)
        y_map = (y_map + dy).astype(np.float32)
        image = cv2.remap(
            image.transpose(1, 2, 0),
            x_map,
            y_map,
            interpolation=cv2.INTER_LINEAR,
        ).transpose(2, 0, 1)
        masks = {
            key: cv2.remap(
                mask.transpose(1, 2, 0),
                x_map,
                y_map,
                interpolation=cv2.INTER_NEAREST,
            ).transpose(2, 0, 1)
            for key, mask in masks.items()
        }
        return {"image": image, "masks": masks, **kwargs}


def train_transform() -> Compose:
    return Compose(
        [
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate90(),
            ElasticDeformation(alpha=25, sigma=4, p=0.3),
            RandomBrightnessContrast(brightness=0.05, contrast=0.1),
            RandomChannelJitter(),
            RandomGamma(),
            RandomGaussianNoise(std=0.01),
            NormalizeTile(),
        ]
    )


def val_transform() -> Compose:
    return Compose([NormalizeTile()])


def test_transform() -> Compose:
    return val_transform()
