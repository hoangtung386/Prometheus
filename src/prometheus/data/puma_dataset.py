from __future__ import annotations

import json
import random
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
        (C, H, W) uint8 mask where C = len(class_map).
    """
    with open(geojson_path) as f:
        data = json.load(f)

    H, W = image_size
    C = len(class_map)
    mask = np.zeros((C, H, W), dtype=np.uint8)

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        raw_label = props.get(label_key, "background")
        class_name = _parse_puma_class_name(raw_label)

        if class_name not in class_map:
            continue
        class_idx = class_map[class_name]

        geometry = feature.get("geometry")
        if geometry is None:
            continue

        for pts in _extract_polygon_coords(geometry):
            cv2.fillPoly(mask[class_idx], [pts], 1)

    return mask


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
        self._mask_cache: dict[str, dict[str, np.ndarray]] = {}

        img_dir = self.root / "images"
        self.image_paths = sorted(img_dir.glob("*.tif"))
        if not self.image_paths:
            self.image_paths = sorted(img_dir.glob("*.png"))
        if not self.image_paths:
            self.image_paths = sorted(img_dir.glob("*.[tT][iI][fF]"))

        print(f"PUMADataset: {len(self.image_paths)} images in {img_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, idx: int) -> np.ndarray:
        path = self.image_paths[idx]

        # tifffile handles TIFFs with various compression
        import tifffile
        img = tifffile.imread(str(path))

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32).transpose(2, 0, 1)
        return img

    def _load_mask(self, idx: int, modality: str) -> np.ndarray:
        img_path = self.image_paths[idx]
        stem = img_path.stem

        cache_key = f"{stem}_{modality}"
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        geojson_dir = self.root / f"geojson_{modality}"
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
            image = np.fliplr(image).copy()
            tissue_mask = np.fliplr(tissue_mask).copy()
            nuclei_mask = np.fliplr(nuclei_mask).copy()

        if random.random() > 0.5:
            image = np.flipud(image).copy()
            tissue_mask = np.flipud(tissue_mask).copy()
            nuclei_mask = np.flipud(nuclei_mask).copy()

        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            tissue_mask = np.rot90(tissue_mask, k, axes=(1, 2)).copy()
            nuclei_mask = np.rot90(nuclei_mask, k, axes=(1, 2)).copy()

        return image, tissue_mask, nuclei_mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        mean = np.mean(image, axis=(1, 2), keepdims=True)
        std = np.std(image, axis=(1, 2), keepdims=True)
        std = np.clip(std, 1e-8, None)
        image = (image - mean) / std

        if self.transforms is not None:
            image = self.transforms(image)

        return image

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images = torch.stack([b[0] for b in batch], dim=0)
        tissue = torch.stack([b[1]["tissue"] for b in batch], dim=0)
        nuclei = torch.stack([b[1]["nuclei"] for b in batch], dim=0)
        return images, {"tissue": tissue, "nuclei": nuclei}

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = self._load_image(idx)
        tissue_mask = self._load_mask(idx, "tissue")
        nuclei_mask = self._load_mask(idx, "nuclei")

        image, tissue_mask, nuclei_mask = self._augment(image, tissue_mask, nuclei_mask)
        image = self._normalize(image)

        targets = {
            "tissue": torch.from_numpy(tissue_mask),
            "nuclei": torch.from_numpy(nuclei_mask),
        }
        return torch.from_numpy(image), targets


def create_puma_dataloaders(
    root: str | Path,
    image_size: int = 1024,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    full_dataset = PUMADataset(root=root, image_size=image_size)

    n = len(full_dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=gen,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=PUMADataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=PUMADataset.collate_fn,
    )
    return train_loader, val_loader
