"""Typed end-to-end predictor for PrometheusNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.spatial import restore_mask
from ..domain import Detection, ImageMeta
from ..models import MultitaskOutput
from .nuclei_decoder import decode_nuclei


@dataclass
class MultitaskPrediction:
    tissue_masks: list[np.ndarray]
    nuclei: list[list[Detection]]


class PrometheusPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        nuclei_stride: int = 4,
        confidence_threshold: float = 0.25,
        max_detections: int = 1000,
        local_max_kernel: int = 3,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.nuclei_stride = nuclei_stride
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.local_max_kernel = local_max_kernel

    @torch.no_grad()
    def predict(self, images: torch.Tensor, metadata: list[ImageMeta]) -> MultitaskPrediction:
        output = self.model(images.to(self.device))
        if not isinstance(output, MultitaskOutput):
            raise TypeError("PrometheusPredictor expects MultitaskOutput")
        masks = output.tissue_logits.argmax(dim=1)
        restored = [restore_mask(mask, meta) for mask, meta in zip(masks, metadata, strict=True)]
        nuclei = decode_nuclei(
            output,
            metadata,
            stride=self.nuclei_stride,
            threshold=self.confidence_threshold,
            max_detections=self.max_detections,
            local_max_kernel=self.local_max_kernel,
        )
        return MultitaskPrediction(restored, nuclei)
