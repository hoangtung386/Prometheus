from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from prometheus.data import PumaMultitaskDataset
from prometheus.domain import NUCLEUS_CLASS_NAMES, TISSUE_CLASS_NAMES, MultitaskSample

_SQUARE = [[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]]


def _write_geojson(path: Path, modality: str) -> None:
    """One 50x50 polygon per class, so a sample exercises the whole taxonomy."""
    names = NUCLEUS_CLASS_NAMES if modality == "nuclei" else TISSUE_CLASS_NAMES[1:]
    features = []
    for class_name in names:
        properties = (
            {"classification": {"name": f"nuclei_{class_name}"}}
            if modality == "nuclei"
            else {"label": class_name.replace("_", " ").title()}
        )
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [_SQUARE]},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def _write_dataset(root: Path) -> None:
    for name in ("images", "geojson_tissue", "geojson_nuclei"):
        (root / name).mkdir()
    tifffile.imwrite(root / "images" / "sample.tif", np.zeros((32, 64, 3), dtype=np.uint8))
    _write_geojson(root / "geojson_tissue" / "sample_tissue.geojson", "tissue")
    _write_geojson(root / "geojson_nuclei" / "sample_nuclei.geojson", "nuclei")


def test_taxonomy_sizes_are_the_checkpoint_contract() -> None:
    assert len(TISSUE_CLASS_NAMES) == 6
    assert len(NUCLEUS_CLASS_NAMES) == 10


def test_dataset_letterboxes_without_distortion_and_keeps_every_instance(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    sample = PumaMultitaskDataset(tmp_path, image_size=(64, 64))[0]

    assert isinstance(sample, MultitaskSample)
    assert sample.image.shape == (3, 64, 64)
    assert sample.metadata.original_size == (32, 64)
    assert sample.metadata.resized_size == (32, 64), "a 2:1 image must not be stretched to square"
    assert sample.metadata.pad_xy == (0, 16)
    assert sample.tissue.mask.shape == (64, 64)


def test_dataset_labels_are_zero_based_over_the_ten_nucleus_classes(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    sample = PumaMultitaskDataset(tmp_path, image_size=(64, 64))[0]

    assert sample.nuclei.centroids.shape == (len(NUCLEUS_CLASS_NAMES), 2)
    assert sample.nuclei.labels.tolist() == list(range(len(NUCLEUS_CLASS_NAMES)))
