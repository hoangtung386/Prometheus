"""Typed samples crossing the data, engine and inference boundaries.

Every batch carries its ``ImageMeta`` so inference can undo the letterbox transform
without re-deriving it, and nuclei stay variable-length per image rather than being padded
into a dense tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ImageMeta", "MultitaskBatch", "MultitaskSample", "NucleiTarget", "TissueTarget"]


@dataclass(frozen=True)
class ImageMeta:
    """The letterbox transform applied to one image, and everything needed to invert it.

    Attributes:
        sample_id: Stable identifier, used to key splits and outputs.
        original_size: ``(height, width)`` of the source image.
        model_size: ``(height, width)`` fed to the model, after padding.
        resized_size: ``(height, width)`` of the content area inside ``model_size``.
        scale_xy: Per-axis resize factor applied before padding.
        pad_xy: ``(left, top)`` padding in model pixels.
    """

    sample_id: str
    original_size: tuple[int, int]
    model_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_xy: tuple[float, float]
    pad_xy: tuple[int, int]


@dataclass
class TissueTarget:
    """Tissue ground truth: a ``(H, W)`` or ``(B, H, W)`` map of training class indices."""

    mask: torch.Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> TissueTarget:
        return TissueTarget(self.mask.to(device, non_blocking=non_blocking))


@dataclass
class NucleiTarget:
    """Nuclei ground truth for one image, as instances.

    Kept variable-length and never rasterized: two touching nuclei of the same class must
    stay two targets, which a semantic mask would merge into one.
    """

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
    """One dataset item: image, tissue mask, nuclei instances and letterbox metadata."""

    image: torch.Tensor
    tissue: TissueTarget
    nuclei: NucleiTarget
    metadata: ImageMeta


@dataclass
class MultitaskBatch:
    """A collated batch. ``nuclei`` stays a list because instance counts differ per image."""

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
