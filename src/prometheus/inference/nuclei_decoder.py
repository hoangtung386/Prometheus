"""Decode dense CenterPoint maps into nuclei detections."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..data.spatial import boxes_to_source, points_to_source
from ..domain import NUCLEUS_TRAIN_ORDER, Detection, ImageMeta
from ..models import MultitaskOutput

__all__ = ["decode_nuclei"]


def _box(values: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = (float(value) for value in values)
    return x_min, y_min, x_max, y_max


def _peak_scores(center_logits: torch.Tensor, local_max_kernel: int) -> torch.Tensor:
    """Keep only local maxima of the center heatmap; this is the NMS of a center head."""
    scores = center_logits.sigmoid()
    pooled = F.max_pool2d(scores, local_max_kernel, stride=1, padding=local_max_kernel // 2)
    return scores * scores.eq(pooled)


def decode_nuclei(
    output: MultitaskOutput,
    metadata: list[ImageMeta] | None = None,
    stride: int = 4,
    threshold: float = 0.25,
    max_detections: int = 1000,
    local_max_kernel: int = 3,
) -> list[list[Detection]]:
    """Turn dense maps into per-image detections.

    Confidence is the product of the class-agnostic peak score and the winning class
    probability. Thresholding that product rather than the peak alone is what keeps
    detection and classification consistent: a confident blob whose class is a coin flip is
    not a confident detection, and the official evaluator ranks candidates by confidence.

    Args:
        output: Dense model maps.
        metadata: Letterbox metadata per image. When given, detections outside the padded
            content area are dropped and the rest are mapped back to source coordinates.
            Pass ``None`` to keep model-space coordinates (useful for visual debugging).
        stride: Spatial stride of the nuclei maps relative to the model input.
        threshold: Minimum combined confidence.
        max_detections: Cap on candidate peaks considered per image, before thresholding.
        local_max_kernel: Odd window size for local-maximum suppression.

    Returns:
        One list of :class:`~prometheus.domain.Detection` per image in the batch.
    """
    if local_max_kernel <= 0 or local_max_kernel % 2 == 0:
        raise ValueError("local_max_kernel must be a positive odd integer")
    if output.nuclei_center_logits.shape[1] != 1:
        raise ValueError(
            "Detection is class-agnostic: nuclei_center_logits must have exactly one channel, "
            f"got {output.nuclei_center_logits.shape[1]}. A class-specific center map produces "
            "one peak per class for the same nucleus (architecture version 1)."
        )

    scores = _peak_scores(output.nuclei_center_logits, local_max_kernel)
    width = scores.shape[-1]
    batch_predictions: list[list[Detection]] = []

    for batch_index in range(scores.shape[0]):
        flat = scores[batch_index].flatten()
        peak_scores, flat_indices = flat.topk(min(max_detections, flat.numel()))
        ys, xs = flat_indices // width, flat_indices % width

        offsets = output.nuclei_offsets[batch_index, :, ys, xs].transpose(0, 1)
        centers = (torch.stack((xs, ys), dim=1).float() + offsets) * stride
        sizes = output.nuclei_sizes[batch_index, :, ys, xs].transpose(0, 1) * stride

        class_probabilities = output.nuclei_class_logits[batch_index, :, ys, xs].softmax(dim=0).transpose(0, 1)
        labels = class_probabilities.argmax(dim=1)
        confidence = peak_scores * class_probabilities.gather(1, labels[:, None]).squeeze(1)

        keep = confidence >= threshold
        confidence, labels = confidence[keep], labels[keep]
        centers, sizes = centers[keep], sizes[keep]
        boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=1)

        if metadata is not None:
            meta = metadata[batch_index]
            inside = _inside_content_area(centers, meta)
            confidence, labels = confidence[inside], labels[inside]
            centers, boxes = centers[inside], boxes[inside]

        center_array = centers.detach().cpu().numpy()
        box_array = boxes.detach().cpu().numpy()
        if metadata is not None:
            center_array = points_to_source(center_array, metadata[batch_index])
            box_array = boxes_to_source(box_array, metadata[batch_index])

        batch_predictions.append(
            [
                Detection(
                    centroid=(float(center_array[index][0]), float(center_array[index][1])),
                    label=NUCLEUS_TRAIN_ORDER[int(labels[index])],
                    confidence=float(confidence[index]),
                    box_xyxy=_box(box_array[index]),
                )
                for index in range(len(confidence))
            ]
        )
    return batch_predictions


def _inside_content_area(centers: torch.Tensor, meta: ImageMeta) -> torch.Tensor:
    """Reject detections that fall in the letterbox padding rather than on tissue."""
    pad_x, pad_y = meta.pad_xy
    height, width = meta.resized_size
    return (
        (centers[:, 0] >= pad_x)
        & (centers[:, 0] < pad_x + width)
        & (centers[:, 1] >= pad_y)
        & (centers[:, 1] < pad_y + height)
    )
