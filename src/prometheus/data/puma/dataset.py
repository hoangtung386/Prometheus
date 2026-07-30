"""Instance-aware PUMA dataset feeding the multitask model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ...domain import (
    NUCLEUS_CLASS_TO_INDEX,
    ImageMeta,
    MultitaskSample,
    NucleiTarget,
    PumaSample,
    TissueTarget,
)
from ...domain.geometry import Polygon
from ..spatial import boxes_to_model, letterbox_image, points_to_model
from ..transforms import TransformSample
from .discovery import discover_puma_samples
from .geojson import parse_nuclei_geojson, parse_tissue_geojson
from .rasterize import rasterize_tissue_regions

__all__ = ["PumaMultitaskDataset", "read_native_image"]

_UINT8_MAX = 255.0
_UINT16_MAX = 65535.0


def read_native_image(path: str | Path) -> np.ndarray:
    """Read a PUMA region of interest as an ``HWC`` float32 array scaled to ``[0, 1]``."""
    import tifffile  # noqa: PLC0415 - heavy optional decoder, imported per read

    image = tifffile.imread(str(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    maximum = float(image.max(initial=0.0))
    if maximum > 1.0:
        image /= _UINT8_MAX if maximum <= _UINT8_MAX else _UINT16_MAX
    return image


def _valid_mask(image_size: tuple[int, int], meta: ImageMeta) -> np.ndarray:
    """Mark the letterboxed content area so padding is excluded from normalization."""
    mask = np.zeros(image_size, dtype=bool)
    pad_x, pad_y = meta.pad_xy
    height, width = meta.resized_size
    mask[pad_y : pad_y + height, pad_x : pad_x + width] = True
    return mask


def _polygon_to_model(polygon: Polygon, meta: ImageMeta) -> Polygon:
    """Map a polygon and all of its interior rings into model coordinates."""
    return Polygon(
        exterior=points_to_model(polygon.exterior, meta),
        holes=tuple(points_to_model(hole, meta) for hole in polygon.holes),
    )


class PumaMultitaskDataset(Dataset[MultitaskSample]):
    """Yield one letterboxed region of interest with its tissue mask and nuclei instances.

    Nuclei are kept as *instances* and never rasterized: two touching nuclei of the same
    class remain two training targets, which a semantic mask would merge into one.

    Args:
        root: PUMA dataset root, as understood by :func:`discover_puma_samples`.
        image_size: Model input size. Images are letterboxed (aspect ratio preserved, then
            padded) and ``ImageMeta`` records the transform so inference can invert it.
        transforms: Synchronised augmentation over image, mask and geometry.
        strict_labels: Raise on an unknown nucleus label instead of skipping it.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] | int = (1024, 1024),
        transforms: Callable[[TransformSample], TransformSample] | None = None,
        strict_labels: bool = True,
    ) -> None:
        self.samples: list[PumaSample] = discover_puma_samples(root)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transforms = transforms
        self.strict_labels = strict_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MultitaskSample:
        sample = self.samples[index]
        image, meta = letterbox_image(read_native_image(sample.image_path), self.image_size, sample.sample_id)

        regions = [
            (label, _polygon_to_model(polygon, meta))
            for label, polygon in parse_tissue_geojson(sample.tissue_annotation_path)
        ]
        tissue_mask = rasterize_tissue_regions(regions, self.image_size)

        instances = parse_nuclei_geojson(sample.nuclei_annotation_path, self.strict_labels)
        centroids = np.asarray([item.centroid for item in instances], dtype=np.float32).reshape(-1, 2)
        boxes = np.asarray([item.box_xyxy for item in instances], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(
            [NUCLEUS_CLASS_TO_INDEX[item.label.value] for item in instances],
            dtype=np.int64,
        )

        transformed = TransformSample(
            image=image.transpose(2, 0, 1),
            tissue_mask=tissue_mask,
            centroids=points_to_model(centroids, meta),
            boxes=boxes_to_model(boxes, meta),
            valid_mask=_valid_mask(self.image_size, meta),
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
