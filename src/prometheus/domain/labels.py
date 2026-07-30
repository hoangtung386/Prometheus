"""Single source of truth for the PUMA taxonomy.

Three distinct index spaces exist and must never be conflated:

``TISSUE_TRAIN_ORDER`` / ``NUCLEUS_TRAIN_ORDER``
    The channel order the model is trained on. **These orders are a checkpoint
    contract**: reordering them silently invalidates every existing checkpoint.
    :mod:`tests.unit.domain.test_labels` pins them.

``TISSUE_SUBMISSION_VALUE``
    The pixel values the challenge expects in the submitted tissue TIFF. Different
    from the training order on purpose; :mod:`prometheus.io.tissue_tiff` remaps.

Track class names
    Track 1 collapses the ten nucleus classes into ``tumor``/``lymphocyte``/``other``;
    Track 2 submits them unchanged. See :func:`nucleus_class_for_track`.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "NUCLEUS_CLASS_NAMES",
    "NUCLEUS_CLASS_TO_INDEX",
    "NUCLEUS_TRAIN_ORDER",
    "TISSUE_CLASS_NAMES",
    "TISSUE_CLASS_TO_INDEX",
    "TISSUE_SUBMISSION_VALUE",
    "TISSUE_TRAIN_ORDER",
    "NucleusClass",
    "TissueClass",
    "Track",
    "normalize_puma_label",
    "nucleus_class_for_track",
    "scored_tissue_classes",
]


class TissueClass(str, Enum):
    """Canonical tissue region labels. ``BACKGROUND`` is not scored by the challenge."""

    BACKGROUND = "background"
    STROMA = "stroma"
    BLOOD_VESSEL = "blood_vessel"
    TUMOR = "tumor"
    EPIDERMIS = "epidermis"
    NECROSIS = "necrosis"


class NucleusClass(str, Enum):
    """Canonical nucleus labels, in the order the ten-class head predicts them."""

    TUMOR = "tumor"
    STROMA = "stroma"
    ENDOTHELIUM = "endothelium"
    HISTIOCYTE = "histiocyte"
    MELANOPHAGE = "melanophage"
    LYMPHOCYTE = "lymphocyte"
    PLASMA_CELL = "plasma_cell"
    NEUTROPHIL = "neutrophil"
    APOPTOSIS = "apoptosis"
    EPITHELIUM = "epithelium"


class Track(str, Enum):
    """Challenge track. Track 1 uses three nucleus classes, Track 2 all ten."""

    TRACK_1 = "track1"
    TRACK_2 = "track2"


# --- Training index spaces (checkpoint contract; do not reorder) ----------------------

TISSUE_TRAIN_ORDER: tuple[TissueClass, ...] = (
    TissueClass.BACKGROUND,
    TissueClass.TUMOR,
    TissueClass.STROMA,
    TissueClass.EPIDERMIS,
    TissueClass.NECROSIS,
    TissueClass.BLOOD_VESSEL,
)
"""Tissue channel order of the segmentation head. Index 0 is the unscored background."""

NUCLEUS_TRAIN_ORDER: tuple[NucleusClass, ...] = tuple(NucleusClass)
"""Nucleus channel order of the classification head. Zero-based, no background class."""

TISSUE_CLASS_NAMES: tuple[str, ...] = tuple(item.value for item in TISSUE_TRAIN_ORDER)
NUCLEUS_CLASS_NAMES: tuple[str, ...] = tuple(item.value for item in NUCLEUS_TRAIN_ORDER)

TISSUE_CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(TISSUE_CLASS_NAMES)}
NUCLEUS_CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(NUCLEUS_CLASS_NAMES)}


def scored_tissue_classes() -> tuple[tuple[str, int], ...]:
    """Return ``(name, train_index)`` for the five tissue classes the challenge scores.

    ``tissue_white_background`` is explicitly excluded from the official micro Dice, so
    every metric and loss that targets the leaderboard must iterate over this tuple
    rather than over all six channels.
    """
    return tuple((name, index) for index, name in enumerate(TISSUE_CLASS_NAMES) if name != TissueClass.BACKGROUND.value)


# --- Submission index space ----------------------------------------------------------

TISSUE_SUBMISSION_VALUE: dict[TissueClass, int] = {
    TissueClass.BACKGROUND: 0,
    TissueClass.STROMA: 1,
    TissueClass.BLOOD_VESSEL: 2,
    TissueClass.TUMOR: 3,
    TissueClass.EPIDERMIS: 4,
    TissueClass.NECROSIS: 5,
}
"""Pixel values required in the submitted tissue mask TIFF."""


# --- Label normalization -------------------------------------------------------------

_ALIASES = {
    "white_background": "background",
    "vascular_endothelium": "endothelium",
    "apoptotic_cell": "apoptosis",
    "apoptotic_cells": "apoptosis",
}
_PREFIXES = ("nuclei_", "nucleus_", "tissue_")


def normalize_puma_label(raw_label: str) -> str:
    """Normalize a raw GeoJSON class name without silently accepting typos.

    Strips the modality prefix, lowercases, unifies separators and resolves the known
    aliases. Unknown names are returned unchanged so that the strict ``TissueClass`` /
    ``NucleusClass`` construction downstream raises instead of guessing.
    """
    normalized = str(raw_label).strip().lower().replace(" ", "_").replace("-", "_")
    for prefix in _PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    return _ALIASES.get(normalized, normalized)


def nucleus_class_for_track(nucleus_class: NucleusClass, track: Track) -> str:
    """Map a canonical nucleus class onto the class name a track expects."""
    if track is Track.TRACK_2:
        return nucleus_class.value
    if nucleus_class is NucleusClass.TUMOR:
        return "tumor"
    if nucleus_class in {NucleusClass.LYMPHOCYTE, NucleusClass.PLASMA_CELL}:
        return "lymphocyte"
    return "other"
