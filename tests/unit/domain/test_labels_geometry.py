import numpy as np

from prometheus.domain import NucleusClass, Track, normalize_puma_label, nucleus_class_for_track
from prometheus.domain.geometry import polygon_centroid, scale_polygon


def test_official_nuclei_labels_are_canonical() -> None:
    assert normalize_puma_label("nuclei_endothelium") == "endothelium"
    assert normalize_puma_label("nuclei_apoptosis") == "apoptosis"
    assert normalize_puma_label("Vascular Endothelium") == "endothelium"


def test_track_one_mapping() -> None:
    assert nucleus_class_for_track(NucleusClass.TUMOR, Track.TRACK_1) == "tumor"
    assert nucleus_class_for_track(NucleusClass.PLASMA_CELL, Track.TRACK_1) == "lymphocyte"
    assert nucleus_class_for_track(NucleusClass.STROMA, Track.TRACK_1) == "other"


def test_scale_polygon_and_centroid() -> None:
    polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    scaled = scale_polygon(polygon, (20, 20), (40, 60))
    assert np.allclose(scaled[-2], [30, 20])
    assert np.allclose(polygon_centroid(scaled), [15, 10])
