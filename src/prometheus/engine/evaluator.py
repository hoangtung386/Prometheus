"""Exact instance-aware validation for the multitask architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.spatial import boxes_to_source, points_to_source
from ..domain import Detection, NucleusClass
from ..inference import decode_nuclei
from ..losses import PrometheusMultitaskLoss
from ..metrics import SegmentationEvaluator, nuclei_detection_metrics


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    tissue_dice: float
    nuclei_macro_f1: float
    metrics: dict[str, float]


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
) -> EvaluationResult:
    model.eval()
    tissue_evaluator = SegmentationEvaluator(6)
    all_predictions, all_targets = [], []
    total_loss = 0.0
    batches = 0
    nucleus_classes = list(NucleusClass)
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        output = model(batch.images)
        losses = criterion(output, batch)
        total_loss += float(losses["total"].item())
        batches += 1
        tissue_evaluator.update(output.tissue_logits, batch.tissue.mask)
        all_predictions.extend(
            decode_nuclei(
                output,
                batch.metadata,
                stride=nuclei_stride,
                threshold=confidence_threshold,
                max_detections=max_detections,
                local_max_kernel=local_max_kernel,
            )
        )
        for target, meta in zip(batch.nuclei, batch.metadata, strict=True):
            centroids = points_to_source(target.centroids.detach().cpu().numpy(), meta)
            boxes = boxes_to_source(target.boxes.detach().cpu().numpy(), meta)
            sample_targets = [
                Detection(
                    centroid=tuple(float(value) for value in centroids[index]),
                    label=nucleus_classes[int(target.labels[index])],
                    box_xyxy=tuple(float(value) for value in boxes[index]),
                )
                for index in range(len(target.labels))
            ]
            all_targets.append(sample_targets)
    tissue_metrics = tissue_evaluator.log_dict("tissue")
    nuclei_metrics = nuclei_detection_metrics(all_predictions, all_targets, nuclei_radius_px)
    tissue_dice = tissue_metrics["tissue/Dice/mean_present_fg"]
    nuclei_f1 = float(nuclei_metrics["macro_f1_summed"])
    return EvaluationResult(
        loss=total_loss / max(batches, 1),
        tissue_dice=tissue_dice,
        nuclei_macro_f1=nuclei_f1,
        metrics={
            **tissue_metrics,
            "nuclei/macro_f1_summed": nuclei_f1,
            "nuclei/macro_f1_per_image": float(nuclei_metrics["macro_f1_per_image"]),
        },
    )
