"""Strict GeoJSON parsing into canonical PUMA domain objects."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...domain import NucleusClass, NucleusInstance, TissueClass, normalize_puma_label
from ...domain.geometry import polygon_box_xyxy, polygon_vertex_mean


def read_geojson(path: str | Path, retries: int = 2, delay: float = 1.0) -> dict[str, Any]:
    geojson_path = Path(path)
    last_error: OSError | None = None
    for attempt in range(retries + 1):
        try:
            with geojson_path.open(encoding="utf-8") as file_obj:
                data = json.load(file_obj)
            if not isinstance(data, dict):
                raise ValueError(f"GeoJSON root must be an object: {geojson_path}")
            return data
        except OSError as error:
            last_error = error
            if attempt < retries:
                time.sleep(delay)
    raise OSError(f"Could not read annotation file {geojson_path}") from last_error


def feature_label(feature: dict[str, Any]) -> str:
    properties = feature.get("properties") or {}
    raw_label = properties.get("label")
    if raw_label is None:
        classification = properties.get("classification") or {}
        if isinstance(classification, dict):
            raw_label = classification.get("name")
    if raw_label is None:
        raise ValueError("GeoJSON feature has no label or classification.name")
    return normalize_puma_label(str(raw_label))


def geometry_polygons(geometry: dict[str, Any] | None) -> list[np.ndarray]:
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
    polygons = []
    for candidate in candidates:
        if not candidate:
            continue
        exterior = np.asarray(candidate[0], dtype=np.float32).reshape(-1, 2)
        if len(exterior) > 1 and np.array_equal(exterior[0], exterior[-1]):
            exterior = exterior[:-1]
        if len(exterior) >= 3:
            polygons.append(exterior)
    return polygons


def parse_nuclei_geojson(path: str | Path, strict: bool = True) -> list[NucleusInstance]:
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
                    polygon=polygon,
                    centroid=polygon_vertex_mean(polygon),
                    box_xyxy=polygon_box_xyxy(polygon),
                )
            )
    return instances


def parse_tissue_geojson(
    path: str | Path,
    strict: bool = True,
) -> list[tuple[TissueClass, np.ndarray]]:
    data = read_geojson(path)
    regions: list[tuple[TissueClass, np.ndarray]] = []
    for feature_index, feature in enumerate(data.get("features", [])):
        try:
            label = TissueClass(feature_label(feature))
        except ValueError:
            if strict:
                raise ValueError(
                    f"Unknown tissue label in {path}, feature {feature_index}: {feature.get('properties')}"
                ) from None
            continue
        for polygon in geometry_polygons(feature.get("geometry")):
            regions.append((label, polygon))
    return regions
