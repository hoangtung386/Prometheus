"""Instance-aware PUMA dataset used by the refactored multitask pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from ...domain import MultitaskSample, NucleiTarget, TissueTarget
from ..spatial import boxes_to_model, letterbox_image, points_to_model
from ..transforms.multitask import TransformSample
from .classes import NUCLEI_CLASS_TO_IDX, TISSUE_CLASS_TO_IDX
from .discovery import discover_puma_samples
from .geojson import parse_nuclei_geojson, parse_tissue_geojson
from .rasterize import rasterize_regions


def read_native_image(path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(str(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    maximum = float(image.max(initial=0.0))
    if maximum > 1.0:
        image /= 255.0 if maximum <= 255.0 else 65535.0
    return image


class PumaMultitaskDataset(Dataset):
    def __init__(
        self,
        root,
        image_size: tuple[int, int] | int = (1024, 1024),
        transforms: Callable[[TransformSample], TransformSample] | None = None,
        strict_labels: bool = True,
    ) -> None:
        self.samples = discover_puma_samples(root)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transforms = transforms
        self.strict_labels = strict_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MultitaskSample:
        sample = self.samples[index]
        image, meta = letterbox_image(read_native_image(sample.image_path), self.image_size, sample.sample_id)
        tissue_regions = [
            (label, points_to_model(polygon, meta))
            for label, polygon in parse_tissue_geojson(sample.tissue_annotation_path)
        ]
        tissue_mask = rasterize_regions(tissue_regions, self.image_size, TISSUE_CLASS_TO_IDX)

        instances = parse_nuclei_geojson(sample.nuclei_annotation_path, self.strict_labels)
        centroids = np.asarray([instance.centroid for instance in instances], dtype=np.float32).reshape(-1, 2)
        boxes = np.asarray([instance.box_xyxy for instance in instances], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray([NUCLEI_CLASS_TO_IDX[instance.label.value] - 1 for instance in instances], dtype=np.int64)
        centroids = points_to_model(centroids, meta)
        boxes = boxes_to_model(boxes, meta)

        valid_mask = np.zeros(self.image_size, dtype=bool)
        pad_x, pad_y = meta.pad_xy
        resized_height, resized_width = meta.resized_size
        valid_mask[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = True
        transformed = TransformSample(
            image.transpose(2, 0, 1),
            tissue_mask,
            centroids,
            boxes,
            valid_mask,
        )
        if self.transforms is not None:
            transformed = self.transforms(transformed)
        return MultitaskSample(
            image=torch.from_numpy(np.ascontiguousarray(transformed.image)).float(),
            tissue=TissueTarget(torch.from_numpy(np.ascontiguousarray(transformed.tissue_mask)).long()),
            nuclei=NucleiTarget(
                centroids=torch.from_numpy(np.ascontiguousarray(transformed.centroids)).float(),
                labels=torch.from_numpy(labels).long(),
                boxes=torch.from_numpy(np.ascontiguousarray(transformed.boxes)).float(),
            ),
            metadata=meta,
        )
