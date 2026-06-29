"""Framework-neutral data structures used at package boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import NucleusClass


@dataclass(frozen=True)
class NucleusInstance:
    instance_id: str
    label: NucleusClass
    polygon: np.ndarray
    centroid: tuple[float, float]
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    centroid: tuple[float, float]
    label: NucleusClass | str
    confidence: float = 1.0
    box_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PumaSample:
    sample_id: str
    image_path: Path
    tissue_annotation_path: Path
    nuclei_annotation_path: Path
