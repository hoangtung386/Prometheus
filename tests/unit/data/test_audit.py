"""Dataset audits. See ``docs/phan-tich-tissue-va-ke-hoach.md`` section 4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from prometheus.data.puma import audit_puma_dataset, audit_resolution, audit_tissue_rasterization

_OUTER = [[0, 0], [400, 0], [400, 400], [0, 400], [0, 0]]
_HOLE = [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]
_NESTED = _HOLE

# cv2.fillPoly includes both boundary rows and columns, so an N-unit square covers
# (N + 1) ** 2 pixels. Spelling that out keeps the expectations below readable.
_OUTER_PIXELS = 401**2
_HOLE_PIXELS = 101**2


def _write_dataset(root: Path, *, tissue_features: list[dict], size: tuple[int, int] = (1024, 1024)) -> None:
    for name in ("images", "geojson_tissue", "geojson_nuclei"):
        (root / name).mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(root / "images" / "sample.tif", np.zeros((*size, 3), dtype=np.uint8))
    (root / "geojson_tissue" / "sample_tissue.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": tissue_features}), encoding="utf-8"
    )
    (root / "geojson_nuclei" / "sample_nuclei.geojson").write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"label": "nuclei_tumor"},
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4]]]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _feature(label: str, rings: list[list[list[int]]]) -> dict:
    return {"properties": {"label": label}, "geometry": {"type": "Polygon", "coordinates": rings}}


def test_integrity_audit_reports_the_class_inventory(tmp_path: Path) -> None:
    _write_dataset(tmp_path, tissue_features=[_feature("tissue_tumor", [_OUTER])])
    report = audit_puma_dataset(tmp_path)

    assert report["sample_count"] == 1
    assert report["tissue_region_counts"] == {"tumor": 1}
    assert report["nuclei_instance_counts"] == {"tumor": 1}
    assert report["errors"] == []


def test_integrity_audit_collects_rather_than_raises_on_a_bad_label(tmp_path: Path) -> None:
    _write_dataset(tmp_path, tissue_features=[_feature("tissue_not_a_class", [_OUTER])])
    report = audit_puma_dataset(tmp_path)

    assert len(report["errors"]) == 1
    assert "Unknown tissue label" in report["errors"][0]["error"]


def test_rasterization_audit_counts_interior_rings(tmp_path: Path) -> None:
    _write_dataset(tmp_path, tissue_features=[_feature("tissue_tumor", [_OUTER, _HOLE])])
    report = audit_tissue_rasterization(tmp_path)

    assert report["interior_ring_count"] == 1
    assert report["images_with_interior_rings"] == 1
    # Dropping the interior ring would hand its whole area to the enclosing class.
    assert report["per_class"]["tumor"]["pixels_correct"] == _OUTER_PIXELS - _HOLE_PIXELS
    assert report["per_class"]["tumor"]["pixels_naive"] == _OUTER_PIXELS


def test_rasterization_audit_quantifies_a_class_erased_by_paint_order(tmp_path: Path) -> None:
    # Nested necrosis listed before the enclosing tumour: naive file-order painting loses it
    # entirely, which is exactly how a rare class ends up absent from the training masks.
    _write_dataset(
        tmp_path,
        tissue_features=[
            _feature("tissue_necrosis", [_NESTED]),
            _feature("tissue_tumor", [_OUTER]),
        ],
    )
    report = audit_tissue_rasterization(tmp_path)
    necrosis = report["per_class"]["necrosis"]

    assert necrosis["pixels_naive"] == 0, "the naive baseline must show the loss"
    assert necrosis["pixels_correct"] == _HOLE_PIXELS
    assert necrosis["images_containing"] == 1
    # A percentage against a zero baseline is meaningless, so the flag carries the finding.
    assert necrosis["absent_from_naive_mask"] is True
    assert necrosis["pixel_delta_percent"] is None


def test_rasterization_audit_reports_images_containing_each_class(tmp_path: Path) -> None:
    _write_dataset(tmp_path, tissue_features=[_feature("tissue_tumor", [_OUTER])])
    report = audit_tissue_rasterization(tmp_path)

    assert report["per_class"]["tumor"]["images_containing"] == 1
    assert report["per_class"]["blood_vessel"]["images_containing"] == 0
    assert report["per_class"]["blood_vessel"]["absent_from_naive_mask"] is False


def test_resolution_audit_accepts_the_native_puma_geometry(tmp_path: Path) -> None:
    _write_dataset(tmp_path, tissue_features=[_feature("tissue_tumor", [_OUTER])])
    report = audit_resolution(tmp_path)

    assert report["image_sizes"] == {"1024x1024": 1}
    assert report["matches_puma_native_size"] is True
    assert report["coordinates_exceed_image"] is False


def test_resolution_audit_flags_a_downsampled_image(tmp_path: Path) -> None:
    _write_dataset(tmp_path, tissue_features=[_feature("tissue_tumor", [_OUTER])], size=(512, 512))
    report = audit_resolution(tmp_path)

    assert report["image_sizes"] == {"512x512": 1}
    assert report["matches_puma_native_size"] is False


def test_resolution_audit_flags_annotations_that_outrun_the_image(tmp_path: Path) -> None:
    # Images downsampled without rescaling the GeoJSON: every mask is misaligned.
    outside = [[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]]
    _write_dataset(tmp_path, tissue_features=[_feature("tissue_tumor", outside)], size=(512, 512))
    report = audit_resolution(tmp_path)

    assert report["coordinate_max"] == 1000.0
    assert report["coordinates_exceed_image"] is True
