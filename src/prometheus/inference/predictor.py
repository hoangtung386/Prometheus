"""Typed end-to-end predictor producing source-space PUMA outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..data.spatial import restore_mask
from ..domain import Detection, ImageMeta
from ..models import MultitaskOutput
from .nuclei_decoder import decode_nuclei
from .tta import DIHEDRAL_VIEWS, tta_forward

__all__ = ["MultitaskPrediction", "PrometheusPredictor"]


@dataclass(frozen=True)
class MultitaskPrediction:
    """Predictions in **source image** coordinates, ready for serialization."""

    tissue_masks: list[np.ndarray]
    nuclei: list[list[Detection]]


class PrometheusPredictor:
    """Run a trained model and undo the letterbox transform.

    Args:
        tta_views: Dihedral views to average over. Defaults to all eight, because the
            submission path is the one place where inference cost does not matter and
            averaging is free accuracy. Pass ``()`` for a single forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        nuclei_stride: int = 4,
        confidence_threshold: float = 0.25,
        max_detections: int = 1000,
        local_max_kernel: int = 3,
        tta_views: Sequence[tuple[int, bool]] = DIHEDRAL_VIEWS,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.nuclei_stride = nuclei_stride
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.local_max_kernel = local_max_kernel
        self.tta_views = tuple(tta_views)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, metadata: list[ImageMeta]) -> MultitaskPrediction:
        """Predict tissue masks and nuclei detections in source-image coordinates."""
        images = images.to(self.device)
        output = tta_forward(self.model, images, self.tta_views) if self.tta_views else self.model(images)
        if not isinstance(output, MultitaskOutput):
            raise TypeError(f"Expected MultitaskOutput, got {type(output).__name__}")
        masks = output.tissue_logits.argmax(dim=1)
        return MultitaskPrediction(
            tissue_masks=[restore_mask(mask, meta) for mask, meta in zip(masks, metadata, strict=True)],
            nuclei=decode_nuclei(
                output,
                metadata,
                stride=self.nuclei_stride,
                threshold=self.confidence_threshold,
                max_detections=self.max_detections,
                local_max_kernel=self.local_max_kernel,
            ),
        )
