"""Encode variable-length nuclei instances for a CenterPoint-style head."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...domain import NucleiTarget


@dataclass
class CenterPointTargets:
    heatmap: torch.Tensor
    indices: list[torch.Tensor]
    labels: list[torch.Tensor]
    offsets: list[torch.Tensor]
    sizes: list[torch.Tensor]


def _draw_gaussian(heatmap: torch.Tensor, center_x: int, center_y: int, radius: int) -> None:
    diameter = radius * 2 + 1
    coordinates = torch.arange(diameter, device=heatmap.device, dtype=heatmap.dtype) - radius
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    gaussian = torch.exp(-(xx.square() + yy.square()) / max(2.0 * (diameter / 6.0) ** 2, 1e-6))
    height, width = heatmap.shape
    left, right = min(center_x, radius), min(width - center_x - 1, radius)
    top, bottom = min(center_y, radius), min(height - center_y - 1, radius)
    if min(left, right, top, bottom) < 0:
        return
    patch = heatmap[center_y - top : center_y + bottom + 1, center_x - left : center_x + right + 1]
    kernel = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    torch.maximum(patch, kernel, out=patch)


def encode_centerpoint_targets(
    targets: list[NucleiTarget],
    output_size: tuple[int, int],
    stride: int,
    num_classes: int,
    gaussian_radius: int = 2,
) -> CenterPointTargets:
    output_height, output_width = output_size
    device = targets[0].centroids.device if targets else torch.device("cpu")
    heatmap = torch.zeros(len(targets), num_classes, output_height, output_width, device=device)
    all_indices, all_labels, all_offsets, all_sizes = [], [], [], []
    for batch_index, target in enumerate(targets):
        scaled = target.centroids / stride
        integer = scaled.floor().long()
        valid = (
            (integer[:, 0] >= 0)
            & (integer[:, 0] < output_width)
            & (integer[:, 1] >= 0)
            & (integer[:, 1] < output_height)
            & (target.labels >= 0)
            & (target.labels < num_classes)
        )
        integer, scaled = integer[valid], scaled[valid]
        labels, boxes = target.labels[valid], target.boxes[valid]
        offsets = scaled - integer.float()
        sizes = torch.stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), dim=1) / stride
        for center, label in zip(integer, labels, strict=True):
            _draw_gaussian(heatmap[batch_index, label], int(center[0]), int(center[1]), gaussian_radius)
        all_indices.append(integer[:, 1] * output_width + integer[:, 0])
        all_labels.append(labels)
        all_offsets.append(offsets)
        all_sizes.append(sizes)
    return CenterPointTargets(heatmap, all_indices, all_labels, all_offsets, all_sizes)
