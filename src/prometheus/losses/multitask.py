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
    def __init__(self, num_nucleus_types: int = 10, output_stride: int = 4, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.num_nucleus_types = num_nucleus_types
        self.output_stride = output_stride
        self.weights = weights or LossWeights()
        self.tissue = MulticlassCombinedLoss(
            ce_weight=self.weights.tissue_ce,
            dice_weight=self.weights.tissue_dice,
        )

    def forward(self, output: MultitaskOutput, batch: MultitaskBatch) -> dict[str, torch.Tensor]:
        tissue_ce, tissue_dice = self.tissue.components(output.tissue_logits, batch.tissue.mask)
        targets = encode_centerpoint_targets(
            batch.nuclei,
            output.nuclei_center_logits.shape[-2:],
            self.output_stride,
            self.num_nucleus_types,
        )
        center = center_focal_loss(output.nuclei_center_logits, targets.heatmap)
        nuclei_class, offset, size = nuclei_regression_losses(
            output.nuclei_class_logits,
            output.nuclei_offsets,
            output.nuclei_sizes,
            targets,
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
