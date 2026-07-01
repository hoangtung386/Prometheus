from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from prometheus.data import PumaMultitaskDataset
from prometheus.data.puma.datasets import (
    NUCLEI_CLASS_TO_IDX,
    NUCLEI_CLASSES,
    TISSUE_CLASS_TO_IDX,
    TISSUE_CLASSES,
    PUMADataset,
)
from prometheus.data.puma.loaders import create_puma_dataloaders
from prometheus.data.puma.rasterize import geojson_to_mask
from prometheus.domain import MultitaskSample


def _make_dummy_geojson(path: Path, modality: str = "tissue") -> None:
    class_map = TISSUE_CLASS_TO_IDX if modality == "tissue" else NUCLEI_CLASS_TO_IDX
    coords = [[[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]]]
    features = []
    for class_name, idx in class_map.items():
        if idx == 0:
            continue
        formatted = f"nuclei_{class_name}" if modality == "nuclei" else class_name.replace("_", " ").title()
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }
        if modality == "nuclei":
            feature["properties"] = {"classification": {"name": formatted}}
        else:
            feature["properties"] = {"label": formatted}
        features.append(feature)
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def test_tissue_classes_count() -> None:
    assert len(TISSUE_CLASSES) == 6


def test_nuclei_classes_count() -> None:
    assert len(NUCLEI_CLASSES) == 11


def test_geojson_to_mask_background_filled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gj = Path(tmp) / "test.geojson"
        _make_dummy_geojson(gj, "tissue")
        mask = geojson_to_mask(gj, (100, 100), TISSUE_CLASS_TO_IDX)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert mask.min() == 0 or mask.max() < 6


def test_geojson_to_mask_empty_geojson() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gj = Path(tmp) / "empty.geojson"
        with open(gj, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
        mask = geojson_to_mask(gj, (64, 64), TISSUE_CLASS_TO_IDX)
        assert mask.shape == (64, 64)
        assert np.all(mask == 0), "All pixels should be background"


def test_puma_dataset_returns_class_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images").mkdir()
        (root / "geojson_tissue").mkdir()
        (root / "geojson_nuclei").mkdir()
        img_path = root / "images" / "dummy.tif"
        np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        import tifffile

        tifffile.imwrite(str(img_path), np_img)
        _make_dummy_geojson(root / "geojson_tissue" / "dummy_tissue.geojson", "tissue")
        _make_dummy_geojson(root / "geojson_nuclei" / "dummy_nuclei.geojson", "nuclei")

        ds = PUMADataset(root=root, image_size=100, augment=False, cache_masks=False)
        img, targets = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 100, 100)
        assert isinstance(targets, dict)
        assert "tissue" in targets and "nuclei" in targets
        assert targets["tissue"].ndim == 2, f"Expected 2D class-index, got {targets['tissue'].shape}"
        assert targets["nuclei"].ndim == 2, f"Expected 2D class-index, got {targets['nuclei'].shape}"
        assert targets["tissue"].dtype == torch.long
        assert targets["nuclei"].dtype == torch.long
        unique_t = targets["tissue"].unique()
        unique_n = targets["nuclei"].unique()
        assert unique_t.min() >= 0 and unique_t.max() < 6
        assert unique_n.min() >= 0 and unique_n.max() < 11


def test_create_puma_dataloaders_stratified_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images").mkdir()
        (root / "geojson_tissue").mkdir()
        (root / "geojson_nuclei").mkdir()
        import tifffile

        for idx in range(4):
            name = f"sample_{idx}"
            np_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            tifffile.imwrite(str(root / "images" / f"{name}.tif"), np_img)
            _make_dummy_geojson(root / "geojson_tissue" / f"{name}_tissue.geojson", "tissue")
            _make_dummy_geojson(root / "geojson_nuclei" / f"{name}_nuclei.geojson", "nuclei")

        train_loader, test_loader = create_puma_dataloaders(
            root=root,
            image_size=32,
            batch_size=1,
            num_workers=0,
            test_split=0.25,
            stratified_split=True,
        )
        assert len(train_loader.dataset) == 3
        assert len(test_loader.dataset) == 1


def test_multitask_dataset_preserves_instances_and_aspect_ratio(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "geojson_tissue").mkdir()
    (tmp_path / "geojson_nuclei").mkdir()
    import tifffile

    tifffile.imwrite(tmp_path / "images" / "sample.tif", np.zeros((32, 64, 3), dtype=np.uint8))
    _make_dummy_geojson(tmp_path / "geojson_tissue" / "sample_tissue.geojson", "tissue")
    _make_dummy_geojson(tmp_path / "geojson_nuclei" / "sample_nuclei.geojson", "nuclei")
    sample = PumaMultitaskDataset(tmp_path, image_size=(64, 64))[0]
    assert isinstance(sample, MultitaskSample)
    assert sample.image.shape == (3, 64, 64)
    assert sample.metadata.original_size == (32, 64)
    assert sample.metadata.resized_size == (32, 64)
    assert sample.metadata.pad_xy == (0, 16)
    assert sample.nuclei.centroids.shape == (10, 2)
    assert sample.nuclei.labels.min() == 0
    assert sample.nuclei.labels.max() == 9
