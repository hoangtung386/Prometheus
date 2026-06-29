"""Deprecated compatibility facade for :mod:`prometheus.data.puma`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain import normalize_puma_label
from .puma.datasets import (
    NUCLEI_CLASS_TO_IDX,
    NUCLEI_CLASSES,
    TISSUE_CLASS_TO_IDX,
    TISSUE_CLASSES,
    PUMADataset,
)
from .puma.geojson import feature_label, geometry_polygons, read_geojson
from .puma.loaders import create_puma_dataloaders
from .puma.rasterize import class_index_to_one_hot, rasterize_regions


def _parse_puma_class_name(raw: str) -> str:
    return normalize_puma_label(raw)


def _extract_polygon_coords(geometry: dict) -> list[np.ndarray]:
    return [polygon.astype(np.int32).reshape(-1, 1, 2) for polygon in geometry_polygons(geometry)]


def _read_geojson(geojson_path: Path, retries: int = 2, delay: float = 1.0) -> dict:
    return read_geojson(geojson_path, retries=retries, delay=delay)


def geojson_to_mask(
    geojson_path: Path,
    image_size: tuple[int, int],
    class_map: dict[str, int],
    label_key: str = "label",
) -> np.ndarray:
    del label_key
    regions = []
    for feature in read_geojson(geojson_path).get("features", []):
        try:
            label = feature_label(feature)
        except ValueError:
            continue
        regions.extend((label, polygon) for polygon in geometry_polygons(feature.get("geometry")))
    labels = rasterize_regions(regions, image_size, class_map)
    return class_index_to_one_hot(labels, len(class_map))


__all__ = [
    "NUCLEI_CLASSES",
    "NUCLEI_CLASS_TO_IDX",
    "PUMADataset",
    "TISSUE_CLASSES",
    "TISSUE_CLASS_TO_IDX",
    "create_puma_dataloaders",
    "geojson_to_mask",
]
