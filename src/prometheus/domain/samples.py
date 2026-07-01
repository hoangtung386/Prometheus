"""Typed samples crossing the data, engine, and inference boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImageMeta:
    sample_id: str
    original_size: tuple[int, int]
    model_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_xy: tuple[float, float]
    pad_xy: tuple[int, int]


@dataclass
class TissueTarget:
    mask: torch.Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> TissueTarget:
        return TissueTarget(self.mask.to(device, non_blocking=non_blocking))


@dataclass
class NucleiTarget:
    centroids: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> NucleiTarget:
        return NucleiTarget(
            centroids=self.centroids.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            boxes=self.boxes.to(device, non_blocking=non_blocking),
        )


@dataclass
class MultitaskSample:
    image: torch.Tensor
    tissue: TissueTarget
    nuclei: NucleiTarget
    metadata: ImageMeta


@dataclass
class MultitaskBatch:
    images: torch.Tensor
    tissue: TissueTarget
    nuclei: list[NucleiTarget]
    metadata: list[ImageMeta]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> MultitaskBatch:
        return MultitaskBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            tissue=self.tissue.to(device, non_blocking=non_blocking),
            nuclei=[target.to(device, non_blocking=non_blocking) for target in self.nuclei],
            metadata=self.metadata,
        )
