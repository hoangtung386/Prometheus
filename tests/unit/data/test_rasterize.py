"""Tissue rasterization: interior rings and overlap priority.

Both properties tested here were absent from an earlier revision, and their absence is the
leading explanation for ``necrosis`` scoring exactly 0.00 on the preliminary test set:
a mask that never contains a class cannot teach it. See
``docs/phan-tich-tissue-va-ke-hoach.md`` section 3.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from prometheus.data.puma import TISSUE_PAINT_PRIORITY, rasterize_tissue_regions
from prometheus.domain import TISSUE_CLASS_TO_INDEX, TissueClass
from prometheus.domain.geometry import Polygon

SIZE = (64, 64)
TUMOR = TISSUE_CLASS_TO_INDEX["tumor"]
STROMA = TISSUE_CLASS_TO_INDEX["stroma"]
NECROSIS = TISSUE_CLASS_TO_INDEX["necrosis"]
BLOOD_VESSEL = TISSUE_CLASS_TO_INDEX["blood_vessel"]


def _square(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def test_empty_input_is_all_background() -> None:
    mask = rasterize_tissue_regions([], SIZE)
    assert mask.shape == SIZE
    assert mask.dtype == np.uint8
    assert not mask.any()


def test_interior_ring_is_punched_out_of_its_own_class() -> None:
    tumor_with_hole = Polygon(exterior=_square(0, 0, 40, 40), holes=(_square(10, 10, 20, 20),))
    mask = rasterize_tissue_regions([(TissueClass.TUMOR, tumor_with_hole)], SIZE)

    assert mask[5, 5] == TUMOR
    assert mask[15, 15] == 0, "the interior ring must not be filled with the enclosing class"


def test_nested_class_survives_an_enclosing_region_listed_after_it() -> None:
    # File order deliberately puts the small nested class first: painting in file order
    # would let the enclosing tumour erase it entirely.
    regions = [
        (TissueClass.NECROSIS, Polygon(_square(10, 10, 20, 20))),
        (TissueClass.BLOOD_VESSEL, Polygon(_square(30, 30, 34, 34))),
        (TissueClass.TUMOR, Polygon(_square(0, 0, 60, 60))),
    ]
    mask = rasterize_tissue_regions(regions, SIZE)

    assert mask[15, 15] == NECROSIS
    assert mask[32, 32] == BLOOD_VESSEL
    assert mask[50, 50] == TUMOR


def test_priority_is_independent_of_input_order() -> None:
    regions = [
        (TissueClass.STROMA, Polygon(_square(0, 0, 60, 60))),
        (TissueClass.BLOOD_VESSEL, Polygon(_square(20, 20, 30, 30))),
    ]
    forward = rasterize_tissue_regions(regions, SIZE)
    reversed_order = rasterize_tissue_regions(list(reversed(regions)), SIZE)

    assert np.array_equal(forward, reversed_order)
    assert forward[25, 25] == BLOOD_VESSEL
    assert forward[5, 5] == STROMA


def test_holes_do_not_erase_a_sibling_polygon_of_the_same_class() -> None:
    # Two tumour features overlap; the first punches a hole where the second is solid.
    # Punching holes into a shared layer instead of a per-feature buffer would lose it.
    regions = [
        (TissueClass.TUMOR, Polygon(exterior=_square(0, 0, 40, 40), holes=(_square(10, 10, 20, 20),))),
        (TissueClass.TUMOR, Polygon(_square(12, 12, 18, 18))),
    ]
    mask = rasterize_tissue_regions(regions, SIZE)

    assert mask[15, 15] == TUMOR
    assert mask[11, 11] == 0, "the part of the hole no sibling covers stays background"


def test_background_never_overwrites_a_scored_class() -> None:
    # Background is unscored, so on an overlap keeping the scored class cannot lose points.
    regions = [
        (TissueClass.TUMOR, Polygon(_square(0, 0, 40, 40))),
        (TissueClass.BACKGROUND, Polygon(_square(0, 0, 60, 60))),
    ]
    assert rasterize_tissue_regions(regions, SIZE)[20, 20] == TUMOR


def test_priority_covers_every_tissue_class() -> None:
    assert set(TISSUE_PAINT_PRIORITY) == set(TissueClass)


def test_class_missing_from_the_priority_is_rejected() -> None:
    regions = [(TissueClass.TUMOR, Polygon(_square(0, 0, 10, 10)))]
    with pytest.raises(ValueError, match="paint priority"):
        rasterize_tissue_regions(regions, SIZE, priority=(TissueClass.STROMA,))
