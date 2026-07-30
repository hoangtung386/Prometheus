"""Stable domain contracts shared across every Prometheus workflow.

``domain`` owns the PUMA taxonomy, pure geometry and the typed samples that cross
package boundaries. It must not import from ``data``, ``models``, ``engine`` or ``io``.
"""

from .geometry import polygon_box_xyxy, polygon_vertex_mean
from .labels import (
    NUCLEUS_CLASS_NAMES,
    NUCLEUS_CLASS_TO_INDEX,
    NUCLEUS_TRAIN_ORDER,
    TISSUE_CLASS_NAMES,
    TISSUE_CLASS_TO_INDEX,
    TISSUE_SUBMISSION_VALUE,
    TISSUE_TRAIN_ORDER,
    NucleusClass,
    TissueClass,
    Track,
    normalize_puma_label,
    nucleus_class_for_track,
    scored_tissue_classes,
)
from .samples import ImageMeta, MultitaskBatch, MultitaskSample, NucleiTarget, TissueTarget
from .types import Detection, NucleusInstance, PumaSample

__all__ = [
    "NUCLEUS_CLASS_NAMES",
    "NUCLEUS_CLASS_TO_INDEX",
    "NUCLEUS_TRAIN_ORDER",
    "TISSUE_CLASS_NAMES",
    "TISSUE_CLASS_TO_INDEX",
    "TISSUE_SUBMISSION_VALUE",
    "TISSUE_TRAIN_ORDER",
    "Detection",
    "ImageMeta",
    "MultitaskBatch",
    "MultitaskSample",
    "NucleiTarget",
    "NucleusClass",
    "NucleusInstance",
    "PumaSample",
    "TissueClass",
    "TissueTarget",
    "Track",
    "normalize_puma_label",
    "nucleus_class_for_track",
    "polygon_box_xyxy",
    "polygon_vertex_mean",
    "scored_tissue_classes",
]
