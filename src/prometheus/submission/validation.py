"""Local structural validation for generated PUMA output files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


def validate_submission_outputs(tissue_path: str | Path, nuclei_path: str | Path) -> None:
    tissue = Path(tissue_path)
    nuclei = Path(nuclei_path)
    if not tissue.is_file() or tissue.stat().st_size == 0:
        raise ValueError(f"Missing or empty tissue output: {tissue}")
    with tifffile.TiffFile(tissue) as tif:
        page = tif.pages[0]
        tissue_mask = page.asarray()
        if tissue_mask.ndim != 2:
            raise ValueError("Tissue output must be a two-dimensional class-index mask")
        if not set(np.unique(tissue_mask)).issubset(set(range(6))):
            raise ValueError("Tissue output contains labels outside the range [0, 5]")
        required_tags = {"XResolution", "YResolution", "SMinSampleValue", "SMaxSampleValue"}
        missing_tags = {tag for tag in required_tags if tag not in page.tags}
        if missing_tags:
            raise ValueError(f"Tissue output is missing TIFF tags: {sorted(missing_tags)}")
    if not nuclei.is_file():
        raise ValueError(f"Missing nuclei output: {nuclei}")
    with nuclei.open(encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    polygons = data.get("polygons")
    if not isinstance(polygons, list):
        raise ValueError("Nuclei output must contain a polygons list")
    for index, polygon in enumerate(polygons):
        if not isinstance(polygon.get("name"), str):
            raise ValueError(f"Polygon {index} has no class name")
        if len(polygon.get("path_points", [])) < 3:
            raise ValueError(f"Polygon {index} must contain at least three path points")
        if not isinstance(polygon.get("score", 1), (int, float)):
            raise ValueError(f"Polygon {index} has an invalid score")
