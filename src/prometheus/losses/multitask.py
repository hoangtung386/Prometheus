"""Explicitly weighted loss composition for PrometheusNet."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn

from ..data.targets import encode_centerpoint_targets
from ..domain import MultitaskBatch
from ..models import MultitaskOutput
from .nuclei import center_focal_loss, nuclei_regression_losses
from .segmentation import MulticlassCombinedLoss

__all__ = ["LossWeights", "PrometheusMultitaskLoss"]


@dataclass(frozen=True)
class LossWeights:
    """Weight of every loss term. Field names must match :class:`~prometheus.config.LossConfig`."""

    tissue_ce: float = 1.0
    tissue_dice: float = 1.0
    center_focal: float = 1.0
    nuclei_class: float = 1.0
    offset: float = 1.0
    size: float = 0.1

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Weight field names, used to project a ``LossConfig`` onto this dataclass."""
        return tuple(field.name for field in fields(cls))


class PrometheusMultitaskLoss(nn.Module):
    """Sum of the tissue and nuclei terms, reporting every component separately.

    Returns raw and weighted values for all six terms so a run can be diagnosed from the
    log alone: a collapsed task shows up as one term stalling, not as an opaque total.
    """

    nuclei_class_weights: torch.Tensor | None

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
        self.tissue = MulticlassCombinedLoss(
            ce_weight=self.weights.tissue_ce,
            dice_weight=self.weights.tissue_dice,
            class_weights=tissue_class_weights,
        )
        self.register_buffer(
            "nuclei_class_weights",
            None if nuclei_class_weights is None else nuclei_class_weights.detach().float(),
            persistent=False,
        )

    def forward(self, output: MultitaskOutput, batch: MultitaskBatch) -> dict[str, torch.Tensor]:
        tissue_ce, tissue_dice = self.tissue.components(output.tissue_logits, batch.tissue.mask)
        targets = encode_centerpoint_targets(
            batch.nuclei,
            (output.nuclei_center_logits.shape[-2], output.nuclei_center_logits.shape[-1]),
            self.output_stride,
            self.num_nucleus_types,
            self.gaussian_radius,
            class_agnostic=True,
        )
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
            "center_focal": center_focal_loss(output.nuclei_center_logits, targets.heatmap),
            "nuclei_class": nuclei_class,
            "offset": offset,
            "size": size,
        }
        weighted = {name: value * getattr(self.weights, name) for name, value in raw.items()}
        return {
            **{f"raw/{name}": value for name, value in raw.items()},
            **{f"weighted/{name}": value for name, value in weighted.items()},
            "total": torch.stack(list(weighted.values())).sum(),
        }
