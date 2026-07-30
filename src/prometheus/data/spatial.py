"""Aspect-ratio preserving spatial transforms and their exact inverses.

The model sees a letterboxed square; the challenge scores the original image. Every
``*_to_model`` function has a ``*_to_source`` inverse, and inference must apply it before
metrics or submission writing. All coordinates are pixel-space ``(x, y)``.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from ..domain import ImageMeta

__all__ = [
    "boxes_to_model",
    "boxes_to_source",
    "letterbox_image",
    "points_to_model",
    "points_to_source",
    "restore_mask",
]


def letterbox_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    sample_id: str,
) -> tuple[np.ndarray, ImageMeta]:
    """Resize an HWC image without distortion and pad it to ``target_size``."""
    source_height, source_width = image.shape[:2]
    target_height, target_width = target_size
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    output = np.zeros((target_height, target_width, resized.shape[2]), dtype=resized.dtype)
    output[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    meta = ImageMeta(
        sample_id=sample_id,
        original_size=(source_height, source_width),
        model_size=(target_height, target_width),
        resized_size=(resized_height, resized_width),
        scale_xy=(resized_width / source_width, resized_height / source_height),
        pad_xy=(pad_x, pad_y),
    )
    return output, meta


def points_to_model(points: np.ndarray, meta: ImageMeta) -> np.ndarray:
    """Map ``[N, 2]`` source-image points into letterboxed model coordinates."""
    transformed = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    transformed[:, 0] = transformed[:, 0] * meta.scale_xy[0] + meta.pad_xy[0]
    transformed[:, 1] = transformed[:, 1] * meta.scale_xy[1] + meta.pad_xy[1]
    return transformed


def points_to_source(points: np.ndarray, meta: ImageMeta) -> np.ndarray:
    """Map ``[N, 2]`` model points back to the source image, clipped to its bounds."""
    transformed = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    transformed[:, 0] = (transformed[:, 0] - meta.pad_xy[0]) / meta.scale_xy[0]
    transformed[:, 1] = (transformed[:, 1] - meta.pad_xy[1]) / meta.scale_xy[1]
    transformed[:, 0] = np.clip(transformed[:, 0], 0, meta.original_size[1] - 1)
    transformed[:, 1] = np.clip(transformed[:, 1], 0, meta.original_size[0] - 1)
    return transformed


def boxes_to_model(boxes: np.ndarray, meta: ImageMeta) -> np.ndarray:
    """Map ``[N, 4]`` xyxy boxes into model coordinates."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.copy()
    corners = boxes.reshape(-1, 2)
    return points_to_model(corners, meta).reshape(-1, 4)


def boxes_to_source(boxes: np.ndarray, meta: ImageMeta) -> np.ndarray:
    """Map ``[N, 4]`` xyxy boxes back to source-image coordinates."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.copy()
    return points_to_source(boxes.reshape(-1, 2), meta).reshape(-1, 4)


def restore_mask(mask: torch.Tensor | np.ndarray, meta: ImageMeta) -> np.ndarray:
    """Crop the padding off a model-space mask and resize it back to the source size.

    Nearest-neighbour resampling, because the values are class indices: interpolating them
    would invent classes that lie numerically between two real ones.
    """
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    pad_x, pad_y = meta.pad_xy
    resized_height, resized_width = meta.resized_size
    cropped = array[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width]
    source_height, source_width = meta.original_size
    return cv2.resize(cropped, (source_width, source_height), interpolation=cv2.INTER_NEAREST)
