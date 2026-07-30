"""Submission validation. Every rejection here is one the challenge would otherwise make
after upload, so the cases are worth covering individually."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from prometheus.domain import TISSUE_CLASS_TO_INDEX, Detection, NucleusClass
from prometheus.io import write_nuclei_json, write_tissue_tiff
from prometheus.submission import validate_submission_outputs


def _valid_pair(tmp_path: Path) -> tuple[Path, Path]:
    tissue_path, nuclei_path = tmp_path / "tissue.tif", tmp_path / "nuclei.json"
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = TISSUE_CLASS_TO_INDEX["tumor"]
    write_tissue_tiff(mask, tissue_path)
    write_nuclei_json(
        [Detection(centroid=(1.0, 2.0), label=NucleusClass.TUMOR, confidence=0.9, box_xyxy=(0.0, 1.0, 2.0, 3.0))],
        nuclei_path,
    )
    return tissue_path, nuclei_path


def test_generated_outputs_validate(tmp_path: Path) -> None:
    validate_submission_outputs(*_valid_pair(tmp_path))


def test_empty_detection_list_is_valid(tmp_path: Path) -> None:
    tissue_path, nuclei_path = tmp_path / "tissue.tif", tmp_path / "nuclei.json"
    write_tissue_tiff(np.zeros((4, 4), dtype=np.uint8), tissue_path)
    write_nuclei_json([], nuclei_path)

    validate_submission_outputs(tissue_path, nuclei_path)


def test_missing_tissue_output_is_rejected(tmp_path: Path) -> None:
    _, nuclei_path = _valid_pair(tmp_path)

    with pytest.raises(ValueError, match="Missing or empty tissue output"):
        validate_submission_outputs(tmp_path / "absent.tif", nuclei_path)


def test_missing_nuclei_output_is_rejected(tmp_path: Path) -> None:
    tissue_path, nuclei_path = _valid_pair(tmp_path)
    nuclei_path.unlink()

    with pytest.raises(ValueError, match="Missing nuclei output"):
        validate_submission_outputs(tissue_path, nuclei_path)


def test_out_of_range_class_index_is_rejected(tmp_path: Path) -> None:
    tissue_path, nuclei_path = _valid_pair(tmp_path)
    tifffile.imwrite(tissue_path, np.full((4, 4), 9, dtype=np.uint8), photometric="minisblack")

    with pytest.raises(ValueError, match=r"outside the range"):
        validate_submission_outputs(tissue_path, nuclei_path)


def test_missing_tiff_tags_are_rejected(tmp_path: Path) -> None:
    # The challenge validator requires the resolution and sample-value tags.
    tissue_path, nuclei_path = _valid_pair(tmp_path)
    tifffile.imwrite(tissue_path, np.zeros((4, 4), dtype=np.uint8), photometric="minisblack")

    with pytest.raises(ValueError, match="missing TIFF tags"):
        validate_submission_outputs(tissue_path, nuclei_path)


def test_non_two_dimensional_mask_is_rejected(tmp_path: Path) -> None:
    tissue_path, nuclei_path = _valid_pair(tmp_path)
    tifffile.imwrite(tissue_path, np.zeros((4, 4, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="two-dimensional"):
        validate_submission_outputs(tissue_path, nuclei_path)


def test_wrong_nuclei_schema_is_rejected(tmp_path: Path) -> None:
    tissue_path, nuclei_path = _valid_pair(tmp_path)
    nuclei_path.write_text(json.dumps({"features": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="polygons"):
        validate_submission_outputs(tissue_path, nuclei_path)


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ({"score": 1.0, "path_points": [[0, 0], [1, 0], [1, 1]]}, "no class name"),
        ({"name": "nuclei_tumor", "path_points": [[0, 0], [1, 0]]}, "three path points"),
        ({"name": "nuclei_tumor", "path_points": [[0, 0], [1, 0], [1, 1]], "score": "high"}, "invalid score"),
    ],
)
def test_malformed_polygons_are_rejected(tmp_path: Path, polygon: dict, message: str) -> None:
    tissue_path, nuclei_path = _valid_pair(tmp_path)
    nuclei_path.write_text(json.dumps({"polygons": [polygon]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_submission_outputs(tissue_path, nuclei_path)
