"""Task-neutral inference pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain import Detection
from .postprocess import semantic_logits_to_detections


@dataclass
class PredictionResult:
    tissue_mask: torch.Tensor
    nuclei: list[list[Detection]]


class PredictionPipeline:
    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> PredictionResult:
        output = self.model(images.to(self.device))
        if not isinstance(output, tuple) or len(output) != 3:
            raise TypeError("PredictionPipeline currently expects a legacy DualUNet output")
        tissue_logits, nuclei_logits, _ = output
        return PredictionResult(
            tissue_mask=tissue_logits.argmax(dim=1).cpu(),
            nuclei=semantic_logits_to_detections(nuclei_logits),
        )
