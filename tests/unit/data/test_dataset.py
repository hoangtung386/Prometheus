from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from prometheus.data.puma_dataset import (
    NUCLEI_CLASS_TO_IDX,
    NUCLEI_CLASSES,
    TISSUE_CLASS_TO_IDX,
    TISSUE_CLASSES,
    PUMADataset,
    create_puma_dataloaders,
    geojson_to_mask,
)


def _make_dummy_geojson(path: Path, modality: str = "tissue") -> None:
    class_map = TISSUE_CLASS_TO_IDX if modality == "tissue" else NUCLEI_CLASS_TO_IDX
    coords = [[[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]]]
    features = []
    for class_name, idx in class_map.items():
        if idx == 0:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"label": class_name.replace("_", " ").title()},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
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
        assert mask.shape == (6, 100, 100)
        assert mask.dtype == np.uint8
        assert np.all(mask.sum(axis=0) == 1), "Every pixel must be exactly one-hot"


def test_geojson_to_mask_nuclei() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gj = Path(tmp) / "test.geojson"
        _make_dummy_geojson(gj, "nuclei")
        mask = geojson_to_mask(gj, (100, 100), NUCLEI_CLASS_TO_IDX)
        assert mask.shape == (11, 100, 100)
        assert np.all(mask.sum(axis=0) == 1)


def test_geojson_to_mask_empty_geojson() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gj = Path(tmp) / "empty.geojson"
        with open(gj, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
        mask = geojson_to_mask(gj, (64, 64), TISSUE_CLASS_TO_IDX)
        assert mask.shape == (6, 64, 64)
        assert np.all(mask[0] == 1), "All pixels should be background"
        assert np.all(mask.sum(axis=0) == 1)


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
