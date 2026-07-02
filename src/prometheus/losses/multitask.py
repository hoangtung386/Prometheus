"""Explicitly weighted loss composition for PrometheusNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.targets import encode_centerpoint_targets
from ..domain import MultitaskBatch
from ..models import MultitaskOutput
from .nuclei import center_focal_loss, nuclei_regression_losses
from .segmentation import MulticlassCombinedLoss


@dataclass(frozen=True)
class LossWeights:
    tissue_ce: float = 1.0
    tissue_dice: float = 1.0
    center_focal: float = 1.0
    nuclei_class: float = 1.0
    offset: float = 1.0
    size: float = 0.1


class PrometheusMultitaskLoss(nn.Module):
    def __init__(
        self,
        num_nucleus_types: int = 10,
        output_stride: int = 4,
        weights: LossWeights | None = None,
        gaussian_radius: int = 2,
        tissue_class_weights: torch.Tensor | None = None,
        nuclei_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if gaussian_radius < 0:
            raise ValueError("gaussian_radius must be non-negative")
        self.num_nucleus_types = num_nucleus_types
        self.output_stride = output_stride
        self.weights = weights or LossWeights()
        self.gaussian_radius = gaussian_radius
        # Per-class weights (inverse frequency) counter the majority-class collapse on the
        # imbalanced PUMA tasks. Registered as buffers so .to(device) relocates them.
        self.tissue = MulticlassCombinedLoss(
            ce_weight=self.weights.tissue_ce,
            dice_weight=self.weights.tissue_dice,
            class_weights=tissue_class_weights,
        )
        if nuclei_class_weights is not None:
            self.register_buffer("nuclei_class_weights", nuclei_class_weights.float())
        else:
            self.nuclei_class_weights = None

    def forward(self, output: MultitaskOutput, batch: MultitaskBatch) -> dict[str, torch.Tensor]:
        tissue_ce, tissue_dice = self.tissue.components(output.tissue_logits, batch.tissue.mask)
        targets = encode_centerpoint_targets(
            batch.nuclei,
            output.nuclei_center_logits.shape[-2:],
            self.output_stride,
            self.num_nucleus_types,
            self.gaussian_radius,
        )
        center = center_focal_loss(output.nuclei_center_logits, targets.heatmap)
        nuclei_class, offset, size = nuclei_regression_losses(
            output.nuclei_class_logits,
            output.nuclei_offsets,
            output.nuclei_sizes,
            targets,
            class_weight=self.nuclei_class_weights,
        )
        raw = {
            "tissue_ce": tissue_ce,
            "tissue_dice": tissue_dice,
            "center_focal": center,
            "nuclei_class": nuclei_class,
            "offset": offset,
            "size": size,
        }
        weighted = {
            name: raw[name] * getattr(self.weights, name)
            for name in raw
        }
        result = {f"raw/{name}": value for name, value in raw.items()}
        result.update({f"weighted/{name}": value for name, value in weighted.items()})
        result["total"] = sum(weighted.values())
        return result
