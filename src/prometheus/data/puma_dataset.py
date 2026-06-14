from __future__ import annotations

import json
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


TISSUE_CLASSES = [
    "background",
    "tumor",
    "stroma",
    "epidermis",
    "necrosis",
    "blood_vessel",
]

NUCLEI_CLASSES = [
    "background",
    "tumor",
    "stroma",
    "vascular_endothelium",
    "histiocyte",
    "melanophage",
    "lymphocyte",
    "plasma_cell",
    "neutrophil",
    "apoptotic_cell",
    "epithelium",
]

TISSUE_CLASS_TO_IDX = {c: i for i, c in enumerate(TISSUE_CLASSES)}
NUCLEI_CLASS_TO_IDX = {c: i for i, c in enumerate(NUCLEI_CLASSES)}


def _parse_puma_class_name(raw: str) -> str:
    raw = raw.strip().lower().replace(" ", "_").replace("-", "_")
    raw = raw.replace("nuclei_", "").replace("tissue_", "")
    return raw


def _extract_polygon_coords(geometry: dict) -> list[np.ndarray]:
    """Extract polygon coordinates from a GeoJSON geometry dict.

    Handles Polygon and MultiPolygon types. Returns list of vertex arrays.
    """
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        return [np.array(coords[0], dtype=np.int32).reshape(-1, 1, 2)]

    if geom_type == "MultiPolygon":
        return [
            np.array(poly[0], dtype=np.int32).reshape(-1, 1, 2)
            for poly in coords
        ]

    return []


def _read_geojson(geojson_path: Path, retries: int = 2, delay: float = 1.0) -> dict:
    last_error: OSError | None = None
    for attempt in range(retries + 1):
        try:
            with open(geojson_path) as f:
                return json.load(f)
        except OSError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay)

    raise OSError(
        f"Could not read annotation file {geojson_path}. "
        "If this path is on Google Drive/Colab and you see "
        "'Transport endpoint is not connected', remount Drive or copy the "
        "dataset to local /content storage before training."
    ) from last_error


def geojson_to_mask(
    geojson_path: Path,
    image_size: tuple[int, int],
    class_map: dict[str, int],
    label_key: str = "label",
) -> np.ndarray:
    """Render GeoJSON polygon annotations to a multi-class mask.

    Args:
        geojson_path: Path to .geojson file.
        image_size: (H, W) of the output mask.
        class_map: Mapping from class name → channel index (0 = background).
        label_key: GeoJSON property key for the class label.

    Returns:
        (C, H, W) uint8 one-hot mask where C = len(class_map). Channel 0 is
        filled as background for pixels not covered by foreground polygons.
    """
    data = _read_geojson(geojson_path)

    H, W = image_size
    C = len(class_map)
    label_mask = np.zeros((H, W), dtype=np.uint8)

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        
        raw_label = props.get(label_key)
        if raw_label is None:
            classification = props.get("classification", {})
            if isinstance(classification, dict):
                raw_label = classification.get("name", "background")
            else:
                raw_label = "background"
        
        class_name = _parse_puma_class_name(raw_label)

        if class_name not in class_map:
            continue
        class_idx = class_map[class_name]

        geometry = feature.get("geometry")
        if geometry is None:
            continue

        if class_idx == 0:
            continue

        for pts in _extract_polygon_coords(geometry):
            cv2.fillPoly(label_mask, [pts], class_idx)

    mask = np.eye(C, dtype=np.uint8)[label_mask].transpose(2, 0, 1)

    foreground = mask[1:].sum(axis=0) > 0
    mask[0] = (~foreground).astype(np.uint8)

    assert np.all(mask.sum(axis=0) == 1), \
        f"Mask must be one-hot per pixel, got sum range [{mask.sum(axis=0).min()}, {mask.sum(axis=0).max()}]"
    return mask


def _presence_labels(dataset: "PUMADataset", idx: int) -> set[str]:
    labels: set[str] = set()
    for modality in ("tissue", "nuclei"):
        mask = dataset._load_mask(idx, modality)
        present = mask.reshape(mask.shape[0], -1).sum(axis=1) > 0
        for cls_idx in np.flatnonzero(present):
            if cls_idx == 0:
                continue
            labels.add(f"{modality}:{cls_idx}")
    return labels


def _random_split_indices(n: int, n_test: int, seed: int) -> tuple[list[int], list[int]]:
    gen = torch.Generator().manual_seed(seed)
    train_split, test_split_indices = torch.utils.data.random_split(
        range(n), [n - n_test, n_test], generator=gen,
    )
    return train_split.indices, test_split_indices.indices


def _stratified_indices(dataset: "PUMADataset", n_test: int, seed: int) -> tuple[list[int], list[int]]:
    n = len(dataset)
    rng = random.Random(seed)
    all_indices = list(range(n))
    rng.shuffle(all_indices)

    sample_labels = {idx: _presence_labels(dataset, idx) for idx in all_indices}
    total_counts: Counter[str] = Counter()
    for labels in sample_labels.values():
        total_counts.update(labels)

    target_counts = {
        label: max(1, round(count * n_test / max(n, 1)))
        for label, count in total_counts.items()
    }
    selected: list[int] = []
    selected_counts: Counter[str] = Counter()
    remaining = set(all_indices)

    while len(selected) < n_test and remaining:
        def score(idx: int) -> tuple[int, int, float]:
            labels = sample_labels[idx]
            unmet = sum(selected_counts[label] < target_counts[label] for label in labels)
            rarity = sum(n - total_counts[label] for label in labels)
            return unmet, rarity, rng.random()

        best = max(remaining, key=score)
        selected.append(best)
        selected_counts.update(sample_labels[best])
        remaining.remove(best)

    test_indices = sorted(selected)
    train_indices = sorted(set(range(n)) - set(test_indices))
    return train_indices, test_indices


class PUMADataset(Dataset):
    """PUMA: Panoptic segmentation of nUclei and tissue in MelanomA.

    Data layout (after unzipping train dataset):
        root/
        ├── images/                    # 01_training_dataset_tif_ROIs/*.tif
        ├── geojson_nuclei/            # 01_training_dataset_geojson_nuclei/*.geojson
        └── geojson_tissue/            # 01_training_dataset_geojson_tissue/*.geojson

    Tissue classes (6):  background, tumor, stroma, epidermis, necrosis, blood_vessel
    Nuclei classes (11): background + tumor, stroma, vascular_endothelium,
                         histiocyte, melanophage, lymphocyte, plasma_cell,
                         neutrophil, apoptotic_cell, epithelium

    Reference: https://puma.grand-challenge.org/dataset/
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 1024,
        task: str = "both",
        augment: bool = True,
        cache_masks: bool = True,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.task = task
        self.augment = augment
        self.cache_masks = cache_masks
        self.transforms = transforms
        self._mask_cache: dict[str, np.ndarray] = {}

        img_dir = self.root / "images"
        self.image_paths = sorted(img_dir.glob("*.tif"))
        if not self.image_paths:
            self.image_paths = sorted(img_dir.glob("*.png"))
        if not self.image_paths:
            self.image_paths = sorted(img_dir.glob("*.[tT][iI][fF]"))

        if not self.image_paths:
            img_dir = self.root / "01_training_dataset_tif_ROIs"
            if img_dir.exists():
                self.image_paths = sorted(img_dir.glob("*.tif"))
                if not self.image_paths:
                    self.image_paths = sorted(img_dir.glob("*.[tT][iI][fF]"))

        print(f"PUMADataset: {len(self.image_paths)} images in {img_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, idx: int) -> np.ndarray:
        path = self.image_paths[idx]

        import tifffile
        img = tifffile.imread(str(path))

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        if img.max() > 1.0:
            scale = 255.0 if img.max() <= 255.0 else 65535.0
            img = img / scale
        img = img.transpose(2, 0, 1)
        return img

    def _load_mask(self, idx: int, modality: str) -> np.ndarray:
        img_path = self.image_paths[idx]
        stem = img_path.stem

        cache_key = f"{stem}_{modality}"
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        geojson_dir = self.root / f"geojson_{modality}"
        if not geojson_dir.exists():
            geojson_dir = self.root / f"01_training_dataset_geojson_{modality}"

        geojson_path = geojson_dir / f"{stem}_{modality}.geojson"

        if not geojson_path.exists():
            geojson_path = geojson_dir / f"{stem}.geojson"

        class_map = TISSUE_CLASS_TO_IDX if modality == "tissue" else NUCLEI_CLASS_TO_IDX
        mask = geojson_to_mask(
            geojson_path=geojson_path,
            image_size=(self.image_size, self.image_size),
            class_map=class_map,
        )

        if self.cache_masks:
            self._mask_cache[cache_key] = mask
        return mask

    def _augment(
        self, image: np.ndarray,
        tissue_mask: np.ndarray,
        nuclei_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.augment:
            return image, tissue_mask, nuclei_mask

        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            tissue_mask = np.flip(tissue_mask, axis=2).copy()
            nuclei_mask = np.flip(nuclei_mask, axis=2).copy()

        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            tissue_mask = np.flip(tissue_mask, axis=1).copy()
            nuclei_mask = np.flip(nuclei_mask, axis=1).copy()

        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            tissue_mask = np.rot90(tissue_mask, k, axes=(1, 2)).copy()
            nuclei_mask = np.rot90(nuclei_mask, k, axes=(1, 2)).copy()

        return image, tissue_mask, nuclei_mask

    def _apply_transforms(
        self, image: np.ndarray, tissue_mask: np.ndarray, nuclei_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.transforms is None:
            mean = np.mean(image, axis=(1, 2), keepdims=True)
            std = np.std(image, axis=(1, 2), keepdims=True)
            std = np.clip(std, 1e-8, None)
            image = (image - mean) / std
            return image, tissue_mask, nuclei_mask

        masks = {"tissue": tissue_mask, "nuclei": nuclei_mask}
        result = self.transforms(image=image, masks=masks)
        image = result["image"]
        tissue_mask = result["masks"]["tissue"]
        nuclei_mask = result["masks"]["nuclei"]
        return image, tissue_mask, nuclei_mask

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images = torch.stack([b[0] for b in batch], dim=0)
        tissue = torch.stack([b[1]["tissue"] for b in batch], dim=0)
        nuclei = torch.stack([b[1]["nuclei"] for b in batch], dim=0)
        return images, {"tissue": tissue, "nuclei": nuclei}

    @staticmethod
    def _one_hot_to_index(mask: np.ndarray) -> np.ndarray:
        return mask.argmax(axis=0).astype(np.int64)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = self._load_image(idx)
        tissue_mask = self._load_mask(idx, "tissue")
        nuclei_mask = self._load_mask(idx, "nuclei")

        image, tissue_mask, nuclei_mask = self._augment(image, tissue_mask, nuclei_mask)
        image, tissue_mask, nuclei_mask = self._apply_transforms(image, tissue_mask, nuclei_mask)

        targets = {
            "tissue": torch.from_numpy(self._one_hot_to_index(tissue_mask)).long(),
            "nuclei": torch.from_numpy(self._one_hot_to_index(nuclei_mask)).long(),
        }
        return torch.from_numpy(image), targets


def create_puma_dataloaders(
    root: str | Path,
    image_size: int = 1024,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
    test_split: Optional[float] = None,
    test_transforms: Optional[Callable] = None,
    stratified_split: bool = True,
) -> tuple[DataLoader, DataLoader]:
    _test_split = test_split if test_split is not None else val_split
    _test_transforms = test_transforms if test_transforms is not None else val_transforms

    train_ds = PUMADataset(
        root=root, image_size=image_size, augment=True,
        transforms=train_transforms,
    )
    test_ds = PUMADataset(
        root=root, image_size=image_size, augment=False,
        transforms=_test_transforms,
    )

    n = len(train_ds)
    n_test = max(1, int(n * _test_split))

    if stratified_split:
        try:
            train_indices, test_indices = _stratified_indices(train_ds, n_test, seed)
        except OSError as exc:
            warnings.warn(
                "Could not build stratified split from annotation files; "
                "falling back to a deterministic random split. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            train_indices, test_indices = _random_split_indices(n, n_test, seed)
    else:
        train_indices, test_indices = _random_split_indices(n, n_test, seed)

    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    test_subset = torch.utils.data.Subset(test_ds, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=PUMADataset.collate_fn,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=PUMADataset.collate_fn,
    )
    return train_loader, test_loader
