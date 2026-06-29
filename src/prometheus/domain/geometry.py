"""Pure geometry helpers using pixel-space ``(x, y)`` coordinates."""

from __future__ import annotations

import numpy as np


def polygon_centroid(polygon: np.ndarray) -> tuple[float, float]:
    """Return a robust centroid for an ``[N, 2]`` polygon."""

    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("Cannot compute the centroid of an empty polygon")
    if len(points) < 3:
        return float(points[:, 0].mean()), float(points[:, 1].mean())

    x = points[:, 0]
    y = points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area_twice = cross.sum()
    if abs(area_twice) < 1e-8:
        return float(x.mean()), float(y.mean())
    cx = ((x + np.roll(x, -1)) * cross).sum() / (3.0 * area_twice)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (3.0 * area_twice)
    return float(cx), float(cy)


def polygon_vertex_mean(polygon: np.ndarray) -> tuple[float, float]:
    """Return the arithmetic vertex mean used by the official PUMA evaluator."""

    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("Cannot compute the vertex mean of an empty polygon")
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def polygon_box_xyxy(polygon: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("Cannot compute a box for an empty polygon")
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def scale_polygon(
    polygon: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale polygon from ``(height, width)`` source to target size."""

    source_height, source_width = source_size
    target_height, target_width = target_size
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Invalid source size: {source_size}")
    scaled = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    scaled[:, 0] *= target_width / source_width
    scaled[:, 1] *= target_height / source_height
    return scaled


def clip_polygon(polygon: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    clipped = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(width - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(height - 1, 0))
    return clipped


def xyxy_to_cxcywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = box
    return (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        x_max - x_min,
        y_max - y_min,
    )
