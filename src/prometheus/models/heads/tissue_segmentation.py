"""Semantic tissue decoder whose depth is independent of the encoder's.

The decoder emits logits at stride 4 and upsamples bilinearly to the input size. That is
ample for the large classes but is a hard ceiling on thin structures: a capillary wall is a
few pixels wide at 40x. See ``docs/phan-tich-tissue-va-ke-hoach.md`` section 3.11d.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...layers import ConvNeXtBlock
from ..contracts import FeaturePyramid

__all__ = ["DecoderLevel", "TissueSegmentationHead"]


class DecoderLevel(nn.Module):
    """One upsampling step: project, add the encoder skip, then refine with ConvNeXt blocks."""

    def __init__(self, input_dim: int, skip_dim: int, output_dim: int, depth: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.skip_projection = nn.Conv2d(skip_dim, output_dim, kernel_size=1)
        self.refinement = nn.Sequential(*[ConvNeXtBlock(output_dim) for _ in range(depth)])

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refinement(self.input_projection(inputs) + self.skip_projection(skip))


class TissueSegmentationHead(nn.Module):
    """Decode the feature pyramid into tissue logits plus a context map for the nuclei branch.

    ``forward`` returns ``(logits, context)``. The context is the stride-8 decoder feature,
    which the gated fusion adds to the nuclei branch on its own grid rather than resampling
    a stride-32 bottleneck.
    """

    def __init__(self, dims: list[int], decoder_depths: list[int], num_classes: int) -> None:
        super().__init__()
        self.s16 = DecoderLevel(dims[3], dims[2], dims[2], decoder_depths[0])
        self.s8 = DecoderLevel(dims[2], dims[1], dims[1], decoder_depths[1])
        self.s4 = DecoderLevel(dims[1], dims[0], dims[0], decoder_depths[2])
        self.classifier = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, features: FeaturePyramid, output_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        decoded_s16 = self.s16(features.s32, features.s16)
        decoded_s8 = self.s8(decoded_s16, features.s8)
        decoded_s4 = self.s4(decoded_s8, features.s4)
        logits = self.classifier(decoded_s4)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits, decoded_s8
