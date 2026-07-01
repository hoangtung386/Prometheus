"""Convert legacy semantic nuclei logits to centroid detections."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from ..data.puma import NUCLEI_CLASSES
from ..domain import Detection, NucleusClass


def semantic_logits_to_detections(logits: torch.Tensor) -> list[list[Detection]]:
    probabilities = logits.softmax(dim=1)
    class_masks = probabilities.argmax(dim=1).detach().cpu().numpy()
    probability_maps = probabilities.detach().cpu().numpy()
    batch_detections = []
    for batch_index, class_mask in enumerate(class_masks):
        detections = []
        for class_index, class_name in enumerate(NUCLEI_CLASSES[1:], start=1):
            count, labels, _, centroids = cv2.connectedComponentsWithStats(
                (class_mask == class_index).astype(np.uint8),
                connectivity=8,
            )
            for component_index in range(1, count):
                component = labels == component_index
                confidence = float(probability_maps[batch_index, class_index][component].mean())
                detections.append(
                    Detection(
                        centroid=tuple(float(value) for value in centroids[component_index]),
                        label=NucleusClass(class_name),
                        confidence=confidence,
                    )
                )
        batch_detections.append(detections)
    return batch_detections


def semantic_targets_to_detections(targets: torch.Tensor) -> list[list[Detection]]:
    """Convert legacy raster targets to centroid targets for detection metrics."""
    if targets.ndim != 3:
        raise ValueError("targets must have shape (B, H, W)")
    batch_detections = []
    for class_mask in targets.detach().cpu().numpy():
        detections = []
        for class_index, class_name in enumerate(NUCLEI_CLASSES[1:], start=1):
            count, _, _, centroids = cv2.connectedComponentsWithStats(
                (class_mask == class_index).astype(np.uint8), connectivity=8
            )
            detections.extend(
                Detection(
                    centroid=tuple(float(value) for value in centroids[index]),
                    label=NucleusClass(class_name),
                )
                for index in range(1, count)
            )
        batch_detections.append(detections)
    return batch_detections
