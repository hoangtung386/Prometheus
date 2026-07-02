"""Convert parsed PUMA regions and instances into semantic masks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from ...domain import NucleusInstance, TissueClass


def rasterize_regions(
    regions: Iterable[tuple[TissueClass | str, np.ndarray]],
    image_size: tuple[int, int],
    class_map: dict[str, int],
) -> np.ndarray:
    height, width = image_size
    label_mask = np.zeros((height, width), dtype=np.uint8)
    for label, polygon in regions:
        name = label.value if hasattr(label, "value") else str(label)
        class_index = class_map.get(name)
        if class_index is None or class_index == 0:
            continue
        points = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(label_mask, [points], class_index)
    return label_mask


def rasterize_instances(
    instances: Iterable[NucleusInstance],
    image_size: tuple[int, int],
    class_map: dict[str, int],
) -> np.ndarray:
    return rasterize_regions(
        ((instance.label.value, instance.polygon) for instance in instances),
        image_size,
        class_map,
    )


def class_index_to_one_hot(label_mask: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.uint8)[label_mask].transpose(2, 0, 1)


def geojson_to_mask(
    geojson_path: str | Path,
    image_size: tuple[int, int],
    class_map: dict[str, int],
) -> np.ndarray:
    from .geojson import parse_tissue_geojson

    regions = [(label, polygon) for label, polygon in parse_tissue_geojson(geojson_path)]
    return rasterize_regions(regions, image_size, class_map)
