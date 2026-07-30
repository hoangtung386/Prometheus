"""ConvNeXt-V2 encoder producing a four-level pyramid shared by both tasks.

Parameter names mirror ``timm``'s ``convnextv2_*`` layout so
:func:`prometheus.models.backbones.pretrained.load_pretrained_backbone` can map weights
tensor-by-tensor. Renaming anything here breaks pretrained loading.
"""

from __future__ import annotations

import torch
from torch import nn

from ...config import PrometheusModelConfig
from ...layers import ConvNeXtBlock, LayerNorm
from ..contracts import FeaturePyramid

__all__ = ["SharedConvNeXtBackbone"]


class SharedConvNeXtBackbone(nn.Module):
    """Stem at stride 4, then four stages; returns features at strides 4, 8, 16 and 32.

    Stochastic depth increases linearly across all blocks, which is the schedule the
    pretrained weights were trained with.
    """

    def __init__(self, config: PrometheusModelConfig) -> None:
        super().__init__()
        dims, depths = config.encoder_dims, config.encoder_depths
        rates = torch.linspace(0, config.drop_path_rate, sum(depths)).tolist()
        offsets = [0]
        for depth in depths[:-1]:
            offsets.append(offsets[-1] + depth)

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ConvNeXtBlock(dim=dims[index], drop_path=rates[offsets[index] + block_index])
                        for block_index in range(depths[index])
                    ]
                )
                for index in range(4)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(dims[index], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
                for index in range(3)
            ]
        )

    def forward(self, images: torch.Tensor) -> FeaturePyramid:
        s4 = self.stages[0](self.stem(images))
        s8 = self.stages[1](self.downsamples[0](s4))
        s16 = self.stages[2](self.downsamples[1](s8))
        s32 = self.stages[3](self.downsamples[2](s16))
        return FeaturePyramid(s4=s4, s8=s8, s16=s16, s32=s32)
