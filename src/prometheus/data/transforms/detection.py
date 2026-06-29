"""Geometry-aware deterministic transforms for nuclei detection."""

from __future__ import annotations

import numpy as np


def horizontal_flip_points(points: np.ndarray, width: int) -> np.ndarray:
    flipped = np.asarray(points, dtype=np.float32).copy()
    flipped[..., 0] = width - 1 - flipped[..., 0]
    return flipped


def vertical_flip_points(points: np.ndarray, height: int) -> np.ndarray:
    flipped = np.asarray(points, dtype=np.float32).copy()
    flipped[..., 1] = height - 1 - flipped[..., 1]
    return flipped
