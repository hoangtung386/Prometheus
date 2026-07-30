"""Production multitask architecture with a genuinely shared shallow representation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import PrometheusModelConfig
from .backbones import SharedConvNeXtBackbone
from .contracts import MultitaskOutput
from .fusion import GatedContextFusion
from .heads import NucleiCenterPointHead, TissueSegmentationHead

__all__ = ["PrometheusNet"]

_ENCODER_STRIDE = 32


class PrometheusNet(nn.Module):
    """Shared ConvNeXt encoder, a semantic tissue decoder and a center-based nuclei head.

    Design decisions worth knowing before changing anything here:

    * Shallow features are genuinely shared; there are not two independent encoders.
    * Tissue context reaches the nuclei branch through a learnable gate on the *same* grid,
      and is not detached. A frozen teacher must be a separate, explicit experiment.
    * Nuclei geometry is predicted at stride 4 or 8, never reconstructed from the stride-32
      semantic bottleneck.
    * Nucleus localization is class-agnostic; the taxonomy is a separate classifier. This
      makes transfer from broad cell detectors possible and prevents one peak per class.

    ``architecture_version`` is written into every checkpoint and checked on load, so an
    incompatible change must bump it.
    """

    architecture_version = 2

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
        """Run both tasks. Input size need not be a multiple of the encoder stride."""
        output_height, output_width = images.shape[-2:]
        padded = F.pad(
            images,
            (0, -output_width % _ENCODER_STRIDE, 0, -output_height % _ENCODER_STRIDE),
        )
        features = self.backbone(padded)

        tissue_logits, tissue_context = self.tissue_head(features, padded.shape[-2:])
        tissue_logits = tissue_logits[..., :output_height, :output_width]

        nuclei_feature = self.nuclei_head.build_feature(features)
        auxiliary: dict[str, torch.Tensor] = {}
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
