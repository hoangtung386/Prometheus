"""Pure geometry helpers and types using pixel-space ``(x, y)`` coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Polygon", "polygon_box_xyxy", "polygon_vertex_mean"]


@dataclass(frozen=True)
class Polygon:
    """A polygon with its interior rings preserved.

    PUMA tissue annotations come from QuPath, where a region containing another region
    (necrosis inside tumour, a vessel inside stroma) is stored as an exterior ring plus
    one or more interior rings. Dropping the interior rings fills the enclosed area with
    the enclosing class and can erase the nested class from the training mask entirely,
    so ``holes`` is part of the contract rather than an optional extra.
    """

    exterior: np.ndarray
    holes: tuple[np.ndarray, ...] = field(default_factory=tuple)


def _as_points(polygon: np.ndarray, operation: str) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError(f"Cannot compute the {operation} of an empty polygon")
    return points


def polygon_vertex_mean(polygon: np.ndarray) -> tuple[float, float]:
    """Return the arithmetic vertex mean, exactly as the official PUMA evaluator does.

    This is deliberately *not* the area-weighted polygon centroid: the evaluator averages
    ``path_points``, so nuclei centroid matching must use the same definition or every
    distance is systematically biased.
    """
    points = _as_points(polygon, "vertex mean")
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def polygon_box_xyxy(polygon: np.ndarray) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box ``(x_min, y_min, x_max, y_max)``."""
    points = _as_points(polygon, "bounding box")
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )
