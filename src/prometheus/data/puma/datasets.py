"""Task-specific and compatibility datasets for PUMA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ...domain import NucleusClass, TissueClass
from ...domain.geometry import polygon_box_xyxy, polygon_vertex_mean, scale_polygon
from .discovery import discover_puma_samples
from .geojson import parse_nuclei_geojson, parse_tissue_geojson
from .rasterize import class_index_to_one_hot, rasterize_instances, rasterize_regions

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
    "endothelium",
    "histiocyte",
    "melanophage",
    "lymphocyte",
    "plasma_cell",
    "neutrophil",
    "apoptosis",
    "epithelium",
]
TISSUE_CLASS_TO_IDX = {name: index for index, name in enumerate(TISSUE_CLASSES)}
NUCLEI_CLASS_TO_IDX = {name: index for index, name in enumerate(NUCLEI_CLASSES)}


def read_image(path: Path, image_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    import tifffile

    image = tifffile.imread(str(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    source_size = image.shape[:2]
    if source_size != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0 if image.max() <= 255.0 else 65535.0
    return image.transpose(2, 0, 1), source_size


def _scale_instances(instances, source_size, target_size):
    scaled = []
    for instance in instances:
        polygon = scale_polygon(instance.polygon, source_size, target_size)
        scaled.append(
            replace(
                instance,
                polygon=polygon,
                centroid=polygon_vertex_mean(polygon),
                box_xyxy=polygon_box_xyxy(polygon),
            )
        )
    return scaled


class PumaTissueDataset(Dataset):
    def __init__(self, root, image_size: int = 1024, transforms: Callable | None = None) -> None:
        self.samples = discover_puma_samples(root)
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image, source_size = read_image(sample.image_path, self.image_size)
        regions = [
            (label, scale_polygon(polygon, source_size, (self.image_size, self.image_size)))
            for label, polygon in parse_tissue_geojson(sample.tissue_annotation_path)
        ]
        mask = rasterize_regions(
            regions,
            (self.image_size, self.image_size),
            TISSUE_CLASS_TO_IDX,
        )
        if self.transforms is not None:
            one_hot = class_index_to_one_hot(mask, len(TISSUE_CLASSES))
            result = self.transforms(image=image, masks={"tissue": one_hot})
            image = result["image"]
            mask = result["masks"]["tissue"].argmax(axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask.astype(np.int64))


class PumaNucleiDataset(Dataset):
    def __init__(self, root, image_size: int = 1024, strict_labels: bool = True) -> None:
        self.samples = discover_puma_samples(root)
        self.image_size = image_size
        self.strict_labels = strict_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image, source_size = read_image(sample.image_path, self.image_size)
        instances = parse_nuclei_geojson(sample.nuclei_annotation_path, self.strict_labels)
        instances = _scale_instances(
            instances,
            source_size,
            (self.image_size, self.image_size),
        )
        boxes = torch.tensor([item.box_xyxy for item in instances], dtype=torch.float32).reshape(-1, 4)
        centroids = torch.tensor([item.centroid for item in instances], dtype=torch.float32).reshape(-1, 2)
        labels = torch.tensor(
            [list(NucleusClass).index(item.label) for item in instances],
            dtype=torch.long,
        )
        return torch.from_numpy(image), {
            "boxes": boxes,
            "centroids": centroids,
            "labels": labels,
            "instances": instances,
            "sample_id": sample.sample_id,
        }


class PUMADataset(Dataset):
    """Compatibility dataset for the legacy dual semantic model."""

    def __init__(
        self,
        root,
        image_size: int = 1024,
        task: str = "both",
        augment: bool = True,
        cache_masks: bool = True,
        transforms: Callable | None = None,
    ) -> None:
        del task
        self.root = Path(root)
        self.samples = discover_puma_samples(root)
        self.image_paths = [sample.image_path for sample in self.samples]
        self.image_size = image_size
        self.augment = augment
        self.cache_masks = cache_masks
        self.transforms = transforms
        self._mask_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, index: int) -> np.ndarray:
        return read_image(self.samples[index].image_path, self.image_size)[0]

    def _load_mask(self, index: int, modality: str) -> np.ndarray:
        sample = self.samples[index]
        cache_key = f"{sample.sample_id}:{modality}"
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        _, source_size = read_image(sample.image_path, self.image_size)
        target_size = (self.image_size, self.image_size)
        if modality == "tissue":
            regions = [
                (label, scale_polygon(polygon, source_size, target_size))
                for label, polygon in parse_tissue_geojson(sample.tissue_annotation_path)
                if label is not TissueClass.BACKGROUND
            ]
            labels = rasterize_regions(regions, target_size, TISSUE_CLASS_TO_IDX)
            class_count = len(TISSUE_CLASSES)
        elif modality == "nuclei":
            instances = _scale_instances(
                parse_nuclei_geojson(sample.nuclei_annotation_path),
                source_size,
                target_size,
            )
            labels = rasterize_instances(instances, target_size, NUCLEI_CLASS_TO_IDX)
            class_count = len(NUCLEI_CLASSES)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        mask = class_index_to_one_hot(labels, class_count)
        if self.cache_masks:
            self._mask_cache[cache_key] = mask
        return mask

    def __getitem__(self, index):
        image = self._load_image(index)
        tissue = self._load_mask(index, "tissue")
        nuclei = self._load_mask(index, "nuclei")
        if self.transforms is not None:
            result = self.transforms(image=image, masks={"tissue": tissue, "nuclei": nuclei})
            image = result["image"]
            tissue = result["masks"]["tissue"]
            nuclei = result["masks"]["nuclei"]
        else:
            mean = image.mean(axis=(1, 2), keepdims=True)
            std = image.std(axis=(1, 2), keepdims=True)
            image = (image - mean) / np.clip(std, 1e-8, None)
        return torch.from_numpy(image), {
            "tissue": torch.from_numpy(tissue.argmax(axis=0).astype(np.int64)),
            "nuclei": torch.from_numpy(nuclei.argmax(axis=0).astype(np.int64)),
        }
