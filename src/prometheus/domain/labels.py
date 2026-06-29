"""Canonical PUMA labels and explicit train/submission mappings."""

from __future__ import annotations

from enum import Enum


class TissueClass(str, Enum):
    BACKGROUND = "background"
    STROMA = "stroma"
    BLOOD_VESSEL = "blood_vessel"
    TUMOR = "tumor"
    EPIDERMIS = "epidermis"
    NECROSIS = "necrosis"


class NucleusClass(str, Enum):
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
    TRACK_1 = "track1"
    TRACK_2 = "track2"


TISSUE_TRAIN_INDEX = {
    TissueClass.BACKGROUND: 0,
    TissueClass.TUMOR: 1,
    TissueClass.STROMA: 2,
    TissueClass.EPIDERMIS: 3,
    TissueClass.NECROSIS: 4,
    TissueClass.BLOOD_VESSEL: 5,
}

TISSUE_SUBMISSION_VALUE = {
    TissueClass.BACKGROUND: 0,
    TissueClass.STROMA: 1,
    TissueClass.BLOOD_VESSEL: 2,
    TissueClass.TUMOR: 3,
    TissueClass.EPIDERMIS: 4,
    TissueClass.NECROSIS: 5,
}

NUCLEUS_TRAIN_INDEX = {nucleus_class: index for index, nucleus_class in enumerate(NucleusClass, start=1)}

TRACK_1_INDEX = {"tumor": 0, "lymphocyte": 1, "other": 2}
TRACK_2_INDEX = {nucleus_class.value: index for index, nucleus_class in enumerate(NucleusClass)}

_ALIASES = {
    "white_background": "background",
    "vascular_endothelium": "endothelium",
    "apoptotic_cell": "apoptosis",
    "apoptotic_cells": "apoptosis",
}


def normalize_puma_label(raw_label: str) -> str:
    """Normalize a raw GeoJSON class name without silently accepting typos."""

    normalized = str(raw_label).strip().lower().replace(" ", "_").replace("-", "_")
    for prefix in ("nuclei_", "nucleus_", "tissue_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return _ALIASES.get(normalized, normalized)


def nucleus_class_for_track(nucleus_class: NucleusClass, track: Track) -> str:
    """Map a canonical nucleus class to a challenge track class name."""

    if track is Track.TRACK_2:
        return nucleus_class.value
    if nucleus_class is NucleusClass.TUMOR:
        return "tumor"
    if nucleus_class in {NucleusClass.LYMPHOCYTE, NucleusClass.PLASMA_CELL}:
        return "lymphocyte"
    return "other"
