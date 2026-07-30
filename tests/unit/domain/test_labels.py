"""Taxonomy contracts.

The training orders below are baked into every checkpoint's channel layout. These
assertions are deliberately literal: reordering a class must fail loudly here rather than
silently mislabel every prediction of an existing checkpoint.
"""

from __future__ import annotations

import pytest

from prometheus.domain import (
    NUCLEUS_CLASS_NAMES,
    NUCLEUS_CLASS_TO_INDEX,
    TISSUE_CLASS_NAMES,
    TISSUE_CLASS_TO_INDEX,
    TISSUE_SUBMISSION_VALUE,
    NucleusClass,
    TissueClass,
    Track,
    normalize_puma_label,
    nucleus_class_for_track,
    scored_tissue_classes,
)


def test_tissue_training_order_is_pinned() -> None:
    assert TISSUE_CLASS_NAMES == ("background", "tumor", "stroma", "epidermis", "necrosis", "blood_vessel")


def test_nucleus_training_order_is_pinned() -> None:
    assert NUCLEUS_CLASS_NAMES == (
        "tumor",
        "stroma",
        "endothelium",
        "histiocyte",
        "melanophage",
        "lymphocyte",
        "plasma_cell",
        "neutrophil",
        "apoptosis",
        "epithelium",
    )


def test_nucleus_indices_are_zero_based_with_no_background_channel() -> None:
    assert NUCLEUS_CLASS_TO_INDEX["tumor"] == 0
    assert len(NUCLEUS_CLASS_TO_INDEX) == 10
    assert "background" not in NUCLEUS_CLASS_TO_INDEX


def test_index_maps_agree_with_the_name_tuples() -> None:
    assert {name: index for index, name in enumerate(TISSUE_CLASS_NAMES)} == TISSUE_CLASS_TO_INDEX
    assert set(TISSUE_CLASS_NAMES) == {item.value for item in TissueClass}
    assert set(NUCLEUS_CLASS_NAMES) == {item.value for item in NucleusClass}


def test_submission_values_differ_from_training_indices() -> None:
    # A silent conflation of these two spaces would relabel every submitted mask.
    assert TISSUE_SUBMISSION_VALUE[TissueClass.STROMA] == 1
    assert TISSUE_SUBMISSION_VALUE[TissueClass.TUMOR] == 3
    assert TISSUE_CLASS_TO_INDEX["tumor"] == 1
    assert set(TISSUE_SUBMISSION_VALUE.values()) == set(range(6))


def test_scored_classes_exclude_background() -> None:
    scored = scored_tissue_classes()

    assert len(scored) == 5
    assert "background" not in [name for name, _ in scored]
    assert all(index == TISSUE_CLASS_TO_INDEX[name] for name, index in scored)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("nuclei_endothelium", "endothelium"),
        ("Vascular Endothelium", "endothelium"),
        ("tissue_white_background", "background"),
        ("apoptotic-cells", "apoptosis"),
        ("  NUCLEI_TUMOR  ", "tumor"),
    ],
)
def test_label_normalization(raw: str, expected: str) -> None:
    assert normalize_puma_label(raw) == expected


def test_unknown_labels_are_passed_through_for_strict_rejection() -> None:
    # Returning the name unchanged lets the enum constructor raise, instead of guessing.
    assert normalize_puma_label("definitely_unknown") == "definitely_unknown"
    with pytest.raises(ValueError, match="definitely_unknown"):
        NucleusClass(normalize_puma_label("definitely_unknown"))


def test_track_one_collapses_ten_classes_into_three() -> None:
    assert nucleus_class_for_track(NucleusClass.TUMOR, Track.TRACK_1) == "tumor"
    assert nucleus_class_for_track(NucleusClass.LYMPHOCYTE, Track.TRACK_1) == "lymphocyte"
    assert nucleus_class_for_track(NucleusClass.PLASMA_CELL, Track.TRACK_1) == "lymphocyte"
    assert nucleus_class_for_track(NucleusClass.STROMA, Track.TRACK_1) == "other"
    assert {nucleus_class_for_track(item, Track.TRACK_1) for item in NucleusClass} == {
        "tumor",
        "lymphocyte",
        "other",
    }


def test_track_two_keeps_the_canonical_names() -> None:
    assert all(nucleus_class_for_track(item, Track.TRACK_2) == item.value for item in NucleusClass)
