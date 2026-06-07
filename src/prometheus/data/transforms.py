from __future__ import annotations

import random

import numpy as np
import cv2


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, **kwargs) -> dict:
        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray, masks: dict[str, np.ndarray], **kwargs) -> dict:
        if random.random() < self.p:
            image = np.flip(image, axis=2).copy()
            masks = {k: np.flip(m, axis=2).copy() for k, m in masks.items()}
        return {"image": image, "masks": masks, **kwargs}


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray, masks: dict[str, np.ndarray], **kwargs) -> dict:
        if random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            masks = {k: np.flip(m, axis=1).copy() for k, m in masks.items()}
        return {"image": image, "masks": masks, **kwargs}


class RandomRotate90:
    def __call__(self, image: np.ndarray, masks: dict[str, np.ndarray], **kwargs) -> dict:
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            masks = {k: np.rot90(m, k, axes=(1, 2)).copy() for k, m in masks.items()}
        return {"image": image, "masks": masks, **kwargs}


class Normalize:
    def __init__(self, mean: tuple[float, ...] | None = None, std: tuple[float, ...] | None = None) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(self.std, dtype=np.float32).reshape(-1, 1, 1)
            image = (image - mean) / std.clip(min=1e-8)
        else:
            mean = image.mean(axis=(1, 2), keepdims=True)
            std = image.std(axis=(1, 2), keepdims=True).clip(min=1e-8)
            image = (image - mean) / std
        return {"image": image, **kwargs}


class NormalizeTile:
    """Per-tile (instance) normalization — good for H&E stained histology."""

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        for c in range(image.shape[0]):
            ch = image[c]
            lo, hi = np.percentile(ch, [2, 98])
            ch = np.clip(ch, lo, hi)
            ch = (ch - ch.mean()) / (ch.std() + 1e-8)
            image[c] = ch
        return {"image": image, **kwargs}


class RandomBrightnessContrast:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        b = random.uniform(-self.brightness, self.brightness)
        c = 1 + random.uniform(-self.contrast, self.contrast)
        image = image * c + b
        return {"image": image, **kwargs}


class RandomGaussianNoise:
    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        noise = np.random.randn(*image.shape).astype(np.float32) * self.std
        image = image + noise
        return {"image": image, **kwargs}


class ElasticDeformation:
    """Elastic deformation for histology images (alpha=sigma control)."""

    def __init__(self, alpha: float = 30, sigma: float = 4, p: float = 0.3) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image: np.ndarray, masks: dict[str, np.ndarray], **kwargs) -> dict:
        if random.random() > self.p:
            return {"image": image, "masks": masks, **kwargs}

        h, w = image.shape[1:]
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1), ksize=(0, 0), sigmaX=self.sigma,
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1), ksize=(0, 0), sigmaX=self.sigma,
        ) * self.alpha

        x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
        x_map = (x_map + dx).astype(np.float32)
        y_map = (y_map + dy).astype(np.float32)

        image = cv2.remap(image.transpose(1, 2, 0), x_map, y_map,
                          interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        masks = {
            k: cv2.remap(m.transpose(1, 2, 0), x_map, y_map,
                         interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
            for k, m in masks.items()
        }
        return {"image": image, "masks": masks, **kwargs}


def train_transform() -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        ElasticDeformation(alpha=25, sigma=4, p=0.3),
        RandomBrightnessContrast(brightness=0.05, contrast=0.1),
        NormalizeTile(),
    ])


def val_transform() -> Compose:
    return Compose([
        NormalizeTile(),
    ])


def collate_puma(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = torch.stack([b[0] for b in batch], dim=0)
    tissue = torch.stack([b[1]["tissue"] for b in batch], dim=0)
    nuclei = torch.stack([b[1]["nuclei"] for b in batch], dim=0)
    return images, {"tissue": tissue, "nuclei": nuclei}
