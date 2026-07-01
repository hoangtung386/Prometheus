"""Production multitask architecture with shared shallow representation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import PrometheusModelConfig
from ..backbones import SharedConvNeXtBackbone
from ..contracts import MultitaskOutput
from ..fusion import GatedContextFusion
from ..heads import NucleiCenterPointHead, TissueSegmentationHead


class PrometheusNet(nn.Module):
    architecture_version = 1

    def __init__(self, config: PrometheusModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrometheusModelConfig()
        self.config.validate()
        dims = self.config.encoder_dims
        self.backbone = SharedConvNeXtBackbone(self.config)
        self.tissue_head = TissueSegmentationHead(
            dims,
            self.config.tissue_decoder_depths,
            self.config.num_tissue_classes,
        )
        self.nuclei_head = NucleiCenterPointHead(
            dims,
            self.config.num_nucleus_types,
            self.config.nuclei_feature_stride,
        )
        nuclei_channels = dims[0] if self.config.nuclei_feature_stride == 4 else dims[1]
        self.context_fusion = GatedContextFusion(nuclei_channels, dims[1]) if self.config.context_enabled else None

    def forward(self, images: torch.Tensor) -> MultitaskOutput:
        output_size = images.shape[-2:]
        pad_height = (-output_size[0]) % 32
        pad_width = (-output_size[1]) % 32
        padded = F.pad(images, (0, pad_width, 0, pad_height))
        features = self.backbone(padded)
        tissue_logits, tissue_context = self.tissue_head(features, padded.shape[-2:])
        tissue_logits = tissue_logits[..., : output_size[0], : output_size[1]]
        nuclei_feature = self.nuclei_head.build_feature(features)
        auxiliary = {}
        if self.context_fusion is not None:
            nuclei_feature, gate_mean = self.context_fusion(nuclei_feature, tissue_context)
            auxiliary["context_gate_mean"] = gate_mean
        centers, classes, offsets, sizes = self.nuclei_head(nuclei_feature)
        return MultitaskOutput(
            tissue_logits=tissue_logits,
            nuclei_center_logits=centers,
            nuclei_class_logits=classes,
            nuclei_offsets=offsets,
            nuclei_sizes=sizes,
            auxiliary=auxiliary,
        )
