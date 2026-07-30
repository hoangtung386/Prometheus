"""Rasterize PUMA tissue polygons into a semantic class-index mask.

Two properties of the PUMA annotations make naive rasterization lossy, and both cost
real leaderboard points on the rare classes:

1. **Interior rings.** A tumour region containing an ulcer, a necrotic focus or a vessel
   is stored as an exterior ring plus interior rings. Filling only the exterior paints
   the enclosing class straight over the nested one.
2. **Overlapping regions.** Tissue features overlap, and the class that survives depends
   entirely on paint order. Painting in file order is arbitrary: a large ``tumor``
   feature listed after a small ``blood_vessel`` feature erases the vessel.

This module therefore composites one binary layer per class (holes punched out) and
resolves overlaps with an explicit, documented priority instead of file order.

Use :func:`prometheus.data.puma.audit.audit_tissue_rasterization` to verify the priority
against a real dataset before changing it.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from ...domain import TISSUE_CLASS_TO_INDEX, TissueClass
from ...domain.geometry import Polygon

__all__ = ["TISSUE_PAINT_PRIORITY", "rasterize_tissue_regions"]

TISSUE_PAINT_PRIORITY: tuple[TissueClass, ...] = (
    TissueClass.BACKGROUND,
    TissueClass.STROMA,
    TissueClass.TUMOR,
    TissueClass.EPIDERMIS,
    TissueClass.NECROSIS,
    TissueClass.BLOOD_VESSEL,
)
"""Paint order for overlapping tissue regions; **later entries win**.

Ordered from the largest, least specific region to the smallest, most specific one, so a
nested structure is never erased by the region that contains it. ``BACKGROUND`` is first
because the challenge does not score ``tissue_white_background``: when it overlaps a
scored class, keeping the scored class cannot lose points, whereas keeping background
would turn the overlap into a false negative.
"""


def _fill(target: np.ndarray, rings: Iterable[np.ndarray], value: int) -> None:
    contours = [np.rint(ring).astype(np.int32).reshape(-1, 1, 2) for ring in rings]
    if contours:
        cv2.fillPoly(target, contours, value)


def rasterize_tissue_regions(
    regions: Iterable[tuple[TissueClass, Polygon]],
    image_size: tuple[int, int],
    priority: tuple[TissueClass, ...] = TISSUE_PAINT_PRIORITY,
) -> np.ndarray:
    """Return a ``(H, W)`` uint8 mask of training class indices.

    Args:
        regions: ``(class, polygon)`` pairs as produced by
            :func:`prometheus.data.puma.geojson.parse_tissue_geojson`.
        image_size: ``(height, width)`` of the output mask.
        priority: Paint order for overlaps; later entries win. Defaults to
            :data:`TISSUE_PAINT_PRIORITY`.

    Unlabelled pixels stay ``0`` (background), which is also what the challenge expects
    for ``tissue_white_background``.
    """
    height, width = image_size
    grouped: dict[TissueClass, list[Polygon]] = {}
    for label, polygon in regions:
        grouped.setdefault(label, []).append(polygon)

    unknown = set(grouped) - set(priority)
    if unknown:
        raise ValueError(f"Tissue classes missing from the paint priority: {sorted(item.value for item in unknown)}")

    mask = np.zeros((height, width), dtype=np.uint8)
    layer = np.zeros((height, width), dtype=np.uint8)
    scratch = np.zeros((height, width), dtype=np.uint8)
    for label in priority:
        polygons = grouped.get(label)
        class_index = TISSUE_CLASS_TO_INDEX[label.value]
        if not polygons or class_index == 0:
            continue
        layer.fill(0)
        for polygon in polygons:
            # A feature-local scratch buffer is required: punching this feature's holes
            # directly into the shared layer would also erase a sibling polygon of the
            # same class that legitimately covers the same pixels.
            scratch.fill(0)
            _fill(scratch, [polygon.exterior], 1)
            _fill(scratch, polygon.holes, 0)
            np.bitwise_or(layer, scratch, out=layer)
        mask[layer.astype(bool)] = class_index
    return mask
