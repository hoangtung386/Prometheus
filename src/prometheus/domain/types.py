"""Framework-neutral data structures used at package boundaries.

Deliberately free of ``torch``: these cross into parsing, metrics and serialization, none of
which should need a deep-learning framework to be tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import NucleusClass

__all__ = ["Detection", "NucleusInstance", "PumaSample"]


@dataclass(frozen=True)
class NucleusInstance:
    """One annotated nucleus.

    ``centroid`` is the arithmetic vertex mean of ``polygon``, matching the official
    evaluator; it is not the area-weighted polygon centroid.
    """

    instance_id: str
    label: NucleusClass
    polygon: np.ndarray
    centroid: tuple[float, float]
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """A predicted or ground-truth nucleus.

    ``confidence`` drives the official matcher's candidate ranking, so ground-truth
    detections keep the default of 1.0.
    """

    centroid: tuple[float, float]
    label: NucleusClass | str
    confidence: float = 1.0
    box_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PumaSample:
    """Paths of one PUMA sample, resolved but not yet read."""

    sample_id: str
    image_path: Path
    tissue_annotation_path: Path
    nuclei_annotation_path: Path
