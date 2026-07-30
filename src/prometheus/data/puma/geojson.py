"""Strict GeoJSON parsing into canonical PUMA domain objects."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...domain import (
    NucleusClass,
    NucleusInstance,
    TissueClass,
    normalize_puma_label,
    polygon_box_xyxy,
    polygon_vertex_mean,
)
from ...domain.geometry import Polygon

__all__ = [
    "feature_label",
    "geometry_polygons",
    "parse_nuclei_geojson",
    "parse_tissue_geojson",
    "read_geojson",
]

_MINIMUM_RING_VERTICES = 3


def read_geojson(path: str | Path, retries: int = 2, delay: float = 1.0) -> dict[str, Any]:
    """Read a GeoJSON document, retrying transient IO errors.

    The retry exists because the supported training workstation reads annotations from a
    network-backed Google Drive mount, where a single ``OSError`` is usually transient.
    """
    geojson_path = Path(path)
    last_error: OSError | None = None
    for attempt in range(retries + 1):
        try:
            with geojson_path.open(encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except OSError as error:
            last_error = error
            if attempt < retries:
                time.sleep(delay)
            continue
        if not isinstance(data, dict):
            raise TypeError(f"GeoJSON root must be an object: {geojson_path}")
        return data
    raise OSError(f"Could not read annotation file {geojson_path}") from last_error


def feature_label(feature: dict[str, Any]) -> str:
    """Return the normalized class name of a GeoJSON feature.

    Accepts both PUMA layouts: a flat ``properties.label`` and QuPath's
    ``properties.classification.name``.
    """
    properties = feature.get("properties") or {}
    raw_label = properties.get("label")
    if raw_label is None:
        classification = properties.get("classification") or {}
        if isinstance(classification, dict):
            raw_label = classification.get("name")
    if raw_label is None:
        raise ValueError("GeoJSON feature has no label or classification.name")
    return normalize_puma_label(str(raw_label))


def _ring(coordinates: Any) -> np.ndarray | None:
    """Return an open ``[N, 2]`` ring, or ``None`` when it is degenerate."""
    ring = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    if len(ring) > 1 and np.array_equal(ring[0], ring[-1]):
        ring = ring[:-1]
    return ring if len(ring) >= _MINIMUM_RING_VERTICES else None


def geometry_polygons(geometry: dict[str, Any] | None) -> list[Polygon]:
    """Return every polygon of a ``Polygon``/``MultiPolygon`` geometry, holes included.

    Non-areal geometry types are skipped rather than raising: PUMA annotations
    occasionally carry point or line features that carry no region information.
    """
    if not geometry:
        return []
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        candidates = [coordinates]
    elif geometry_type == "MultiPolygon":
        candidates = coordinates
    else:
        return []

    polygons: list[Polygon] = []
    for candidate in candidates:
        if not candidate:
            continue
        exterior = _ring(candidate[0])
        if exterior is None:
            continue
        holes = tuple(ring for ring in (_ring(item) for item in candidate[1:]) if ring is not None)
        polygons.append(Polygon(exterior=exterior, holes=holes))
    return polygons


def parse_nuclei_geojson(path: str | Path, strict: bool = True) -> list[NucleusInstance]:
    """Parse nuclei annotations, keeping touching same-class nuclei as separate instances.

    Only the exterior ring is used: the official evaluator matches on the arithmetic mean
    of a nucleus outline, and nuclei are not annotated with interior rings.
    """
    data = read_geojson(path)
    instances: list[NucleusInstance] = []
    for feature_index, feature in enumerate(data.get("features", [])):
        try:
            label = NucleusClass(feature_label(feature))
        except ValueError:
            if strict:
                raise ValueError(
                    f"Unknown nuclei label in {path}, feature {feature_index}: {feature.get('properties')}"
                ) from None
            continue
        polygons = geometry_polygons(feature.get("geometry"))
        if not polygons and strict:
            raise ValueError(f"Invalid nuclei geometry in {path}, feature {feature_index}")
        feature_id = str(feature.get("id", feature_index))
        for polygon_index, polygon in enumerate(polygons):
            instance_id = feature_id if len(polygons) == 1 else f"{feature_id}:{polygon_index}"
            instances.append(
                NucleusInstance(
                    instance_id=instance_id,
                    label=label,
                    polygon=polygon.exterior,
                    centroid=polygon_vertex_mean(polygon.exterior),
                    box_xyxy=polygon_box_xyxy(polygon.exterior),
                )
            )
    return instances


def parse_tissue_geojson(
    path: str | Path,
    strict: bool = True,
) -> list[tuple[TissueClass, Polygon]]:
    """Parse tissue annotations into ``(class, polygon-with-holes)`` pairs."""
    data = read_geojson(path)
    regions: list[tuple[TissueClass, Polygon]] = []
    for feature_index, feature in enumerate(data.get("features", [])):
        try:
            label = TissueClass(feature_label(feature))
        except ValueError:
            if strict:
                raise ValueError(
                    f"Unknown tissue label in {path}, feature {feature_index}: {feature.get('properties')}"
                ) from None
            continue
        regions.extend((label, polygon) for polygon in geometry_polygons(feature.get("geometry")))
    return regions
