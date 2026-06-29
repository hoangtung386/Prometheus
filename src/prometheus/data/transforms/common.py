"""Photometric transforms shared by segmentation and detection tasks."""

from __future__ import annotations

import random

import numpy as np


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, **kwargs) -> dict:
        for transform in self.transforms:
            kwargs = transform(**kwargs)
        return kwargs


class Normalize:
    def __init__(self, mean=None, std=None) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if self.mean is None or self.std is None:
            mean = image.mean(axis=(1, 2), keepdims=True)
            std = image.std(axis=(1, 2), keepdims=True)
        else:
            mean = np.asarray(self.mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.asarray(self.std, dtype=np.float32).reshape(-1, 1, 1)
        return {"image": (image - mean) / np.clip(std, 1e-8, None), **kwargs}


class NormalizeTile:
    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        image = image.copy()
        for channel_index in range(image.shape[0]):
            channel = image[channel_index]
            low, high = np.percentile(channel, [2, 98])
            channel = np.clip(channel, low, high)
            image[channel_index] = (channel - channel.mean()) / (channel.std() + 1e-8)
        return {"image": image, **kwargs}


class RandomBrightnessContrast:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        brightness = random.uniform(-self.brightness, self.brightness)
        contrast = 1.0 + random.uniform(-self.contrast, self.contrast)
        return {"image": np.clip(image * contrast + brightness, 0.0, 1.0), **kwargs}


class RandomGaussianNoise:
    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        noise = np.random.randn(*image.shape).astype(np.float32) * self.std
        return {"image": np.clip(image + noise, 0.0, 1.0), **kwargs}


class RandomChannelJitter:
    def __init__(self, scale: float = 0.08, shift: float = 0.03, p: float = 0.8) -> None:
        self.scale = scale
        self.shift = shift
        self.p = p

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if random.random() > self.p:
            return {"image": image, **kwargs}
        shape = (image.shape[0], 1, 1)
        gains = np.random.uniform(1 - self.scale, 1 + self.scale, size=shape).astype(np.float32)
        shifts = np.random.uniform(-self.shift, self.shift, size=shape).astype(np.float32)
        return {"image": np.clip(image * gains + shifts, 0.0, 1.0), **kwargs}


class RandomGamma:
    def __init__(self, gamma_range=(0.8, 1.25), p: float = 0.5) -> None:
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if random.random() > self.p:
            return {"image": image, **kwargs}
        gamma = random.uniform(*self.gamma_range)
        return {"image": np.clip(image, 0.0, 1.0) ** gamma, **kwargs}
