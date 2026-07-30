"""Instance-aware validation reporting the two official challenge metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.spatial import boxes_to_source, points_to_source
from ..domain import NUCLEUS_TRAIN_ORDER, TISSUE_CLASS_NAMES, Detection, MultitaskBatch
from ..inference import decode_nuclei
from ..inference.tta import tta_forward
from ..losses import PrometheusMultitaskLoss
from ..metrics import SegmentationEvaluator, nuclei_detection_metrics

__all__ = ["EvaluationResult", "evaluate_multitask"]


@dataclass(frozen=True)
class EvaluationResult:
    """Validation outcome. ``metrics`` carries the per-class breakdown for logging."""

    loss: float
    tissue_micro_dice: float
    nuclei_macro_f1: float
    metrics: dict[str, float] = field(default_factory=dict)


def _box(values: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = (float(value) for value in values)
    return x_min, y_min, x_max, y_max


def _targets_in_source_space(batch: MultitaskBatch) -> list[list[Detection]]:
    """Map ground-truth nuclei back to original image coordinates for exact matching."""
    per_image: list[list[Detection]] = []
    for target, meta in zip(batch.nuclei, batch.metadata, strict=True):
        centroids = points_to_source(target.centroids.detach().cpu().numpy(), meta)
        boxes = boxes_to_source(target.boxes.detach().cpu().numpy(), meta)
        per_image.append(
            [
                Detection(
                    centroid=(float(centroids[index][0]), float(centroids[index][1])),
                    label=NUCLEUS_TRAIN_ORDER[int(target.labels[index])],
                    box_xyxy=_box(boxes[index]),
                )
                for index in range(len(target.labels))
            ]
        )
    return per_image


@torch.no_grad()
def evaluate_multitask(
    model: torch.nn.Module,
    loader,
    criterion: PrometheusMultitaskLoss,
    device: torch.device,
    nuclei_stride: int = 4,
    nuclei_radius_px: float = 15.0,
    confidence_threshold: float = 0.25,
    max_detections: int = 1000,
    local_max_kernel: int = 3,
    tta_views: Sequence[tuple[int, bool]] | None = None,
) -> EvaluationResult:
    """Evaluate official tissue micro Dice and nuclei centroid F1 on a validation loader.

    Detections are restored to source-image coordinates before matching, so the F1 is
    computed where the official evaluator computes it rather than in the letterboxed frame.

    Args:
        tta_views: Dihedral views to average over, or ``None`` for a single forward pass.
            The loss is always measured on the plain pass; TTA only refines the metrics.
    """
    model.eval()
    tissue_evaluator = SegmentationEvaluator(class_names=TISSUE_CLASS_NAMES)
    predictions: list[list[Detection]] = []
    targets: list[list[Detection]] = []
    total_loss = 0.0
    batches = 0

    for raw_batch in loader:
        batch = raw_batch.to(device, non_blocking=True)
        output = model(batch.images)
        total_loss += float(criterion(output, batch)["total"])
        batches += 1

        scored = tta_forward(model, batch.images, tta_views) if tta_views else output
        tissue_evaluator.update(scored.tissue_logits, batch.tissue.mask)
        predictions.extend(
            decode_nuclei(
                scored,
                batch.metadata,
                stride=nuclei_stride,
                threshold=confidence_threshold,
                max_detections=max_detections,
                local_max_kernel=local_max_kernel,
            )
        )
        targets.extend(_targets_in_source_space(batch))

    tissue_metrics = tissue_evaluator.log_dict("tissue")
    nuclei_metrics = nuclei_detection_metrics(predictions, targets, nuclei_radius_px)
    nuclei_f1 = nuclei_metrics.macro_f1_summed
    return EvaluationResult(
        loss=total_loss / max(batches, 1),
        tissue_micro_dice=tissue_metrics["tissue/micro_dice"],
        nuclei_macro_f1=nuclei_f1,
        metrics={
            **tissue_metrics,
            "nuclei/macro_f1_summed": nuclei_f1,
            "nuclei/macro_f1_per_image": nuclei_metrics.macro_f1_per_image,
        },
    )
