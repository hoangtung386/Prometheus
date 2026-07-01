"""Decode CenterPoint maps into domain detections."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.spatial import boxes_to_source, points_to_source
from ..domain import Detection, ImageMeta, NucleusClass
from ..models import MultitaskOutput


def decode_nuclei(
    output: MultitaskOutput,
    metadata: list[ImageMeta] | None = None,
    stride: int = 4,
    threshold: float = 0.25,
    max_detections: int = 1000,
    local_max_kernel: int = 3,
) -> list[list[Detection]]:
    scores = output.nuclei_center_logits.sigmoid()
    pooled = F.max_pool2d(scores, local_max_kernel, stride=1, padding=local_max_kernel // 2)
    scores = scores * scores.eq(pooled)
    batch_predictions = []
    classes = list(NucleusClass)
    for batch_index in range(scores.shape[0]):
        flat = scores[batch_index].flatten()
        count = min(max_detections, flat.numel())
        values, flat_indices = flat.topk(count)
        keep = values >= threshold
        values, flat_indices = values[keep], flat_indices[keep]
        height, width = scores.shape[-2:]
        spatial_indices = flat_indices % (height * width)
        ys, xs = spatial_indices // width, spatial_indices % width
        offsets = output.nuclei_offsets[batch_index, :, ys, xs].transpose(0, 1)
        centers = torch.stack((xs, ys), dim=1).float() + offsets
        centers = centers * stride
        class_probabilities = output.nuclei_class_logits[batch_index, :, ys, xs].softmax(dim=0).transpose(0, 1)
        predicted_classes = class_probabilities.argmax(dim=1)
        class_confidence = class_probabilities.gather(1, predicted_classes[:, None]).squeeze(1)
        labels = predicted_classes
        sizes = output.nuclei_sizes[batch_index, :, ys, xs].transpose(0, 1) * stride
        boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=1)
        if metadata is not None:
            meta = metadata[batch_index]
            pad_x, pad_y = meta.pad_xy
            resized_height, resized_width = meta.resized_size
            valid = (
                (centers[:, 0] >= pad_x)
                & (centers[:, 0] < pad_x + resized_width)
                & (centers[:, 1] >= pad_y)
                & (centers[:, 1] < pad_y + resized_height)
            )
            values = values[valid]
            class_confidence = class_confidence[valid]
            labels = labels[valid]
            centers = centers[valid]
            boxes = boxes[valid]
        center_array = centers.detach().cpu().numpy()
        box_array = boxes.detach().cpu().numpy()
        if metadata is not None:
            center_array = points_to_source(center_array, metadata[batch_index])
            box_array = boxes_to_source(box_array, metadata[batch_index])
        detections = []
        for index in range(len(values)):
            class_index = int(labels[index])
            if class_index >= len(classes):
                continue
            detections.append(
                Detection(
                    centroid=tuple(float(value) for value in center_array[index]),
                    label=classes[class_index],
                    confidence=float((values[index] * class_confidence[index]).item()),
                    box_xyxy=tuple(float(value) for value in box_array[index]),
                )
            )
        batch_predictions.append(detections)
    return batch_predictions
