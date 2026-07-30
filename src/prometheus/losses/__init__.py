"""Tissue, nuclei and multitask loss composition."""

from .class_weights import (
    BACKGROUND_TISSUE_WEIGHT,
    class_weights_from_counts,
    compute_class_weights,
)
from .multitask import LossWeights, PrometheusMultitaskLoss
from .nuclei import center_focal_loss, nuclei_regression_losses
from .segmentation import MulticlassCombinedLoss, MultiClassDiceLoss

__all__ = [
    "BACKGROUND_TISSUE_WEIGHT",
    "LossWeights",
    "MultiClassDiceLoss",
    "MulticlassCombinedLoss",
    "PrometheusMultitaskLoss",
    "center_focal_loss",
    "class_weights_from_counts",
    "compute_class_weights",
    "nuclei_regression_losses",
]
