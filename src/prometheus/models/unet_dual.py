from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..blocks import LocalGlobalAttention
from ..config import ModelConfig
from ..utils import LayerNorm
from ._base_unet import build_decoder, build_encoder


class TissueAttentionEncoder(nn.Module):
    """Encodes tissue feature map into bottleneck-level features for cross-attention."""

    def __init__(self, in_channels: int, dims: List[int]) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.downsample_layers:
            x = layer(x)
        return x


class TissueDecoder(nn.Module):
    """Decoder that returns both mask and feature map."""

    def __init__(self, encoder_dims: List[int], encoder_depths: List[int], num_classes: int = 1) -> None:
        super().__init__()
        self.levels, _ = build_decoder(encoder_dims, encoder_depths)

        self.feature_head = nn.ConvTranspose2d(encoder_dims[0], encoder_dims[0], kernel_size=4, stride=4)
        self.mask_head = nn.Conv2d(encoder_dims[0], num_classes, kernel_size=1)

    def forward(
        self, x: torch.Tensor, skips: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for level in range(3):
            stage_skips = skips[2 - level]
            for j, layer in enumerate(self.levels[level]):
                x = layer(x, stage_skips[j])
        features = self.feature_head(x)
        mask = self.mask_head(features)
        return mask, features


class DualUNet(nn.Module):
    """Unified dual-stream architecture for tissue and nuclei segmentation.

    Tissue stream produces a tissue mask. Its feature map is then encoded
    (with stop-gradient) and fused into the nuclei stream via cross-attention.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()

        dims = config.encoder_dims
        depths = config.encoder_depths

        # === TISSUE STREAM ===
        self.tissue_stem, self.tissue_down, self.tissue_stages = build_encoder(
            in_chans=config.in_chans,
            dims=dims,
            depths=depths,
            drop_path_rate=config.drop_path_rate,
        )
        self.tissue_decoder = TissueDecoder(
            encoder_dims=dims,
            encoder_depths=depths,
            num_classes=config.num_classes,
        )

        # === TISSUE → NUCLEI BRIDGE ===
        self.tissue_attention_encoder = TissueAttentionEncoder(
            in_channels=dims[0],
            dims=dims,
        )
        self.cross_attention = LocalGlobalAttention(
            d_model=dims[-1], n_heads=8, window_size=2
        )

        # === NUCLEI STREAM ===
        self.nuclei_stem, self.nuclei_down, self.nuclei_stages = build_encoder(
            in_chans=config.in_chans,
            dims=dims,
            depths=depths,
            drop_path_rate=config.drop_path_rate,
        )
        self.nuclei_decoder = TissueDecoder(
            encoder_dims=dims,
            encoder_depths=depths,
            num_classes=config.num_classes,
        )

    def _encode(self, x, stem, down, stages):
        """Run forward_encoder logic: returns (bottleneck, skip_connections)."""
        all_skips: List[List[torch.Tensor]] = []

        x = stem(x)
        stage_skips: List[torch.Tensor] = []
        for layer in stages[0]:
            x = layer(x)
            stage_skips.append(x)
        all_skips.append(stage_skips)

        for i in range(3):
            x = down[i](x)
            stage_skips = []
            for layer in stages[i + 1]:
                x = layer(x)
                stage_skips.append(x)
            all_skips.append(stage_skips)

        return x, all_skips[:-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # === TISSUE STREAM ===
        tissue_bottleneck, tissue_skips = self._encode(
            x, self.tissue_stem, self.tissue_down, self.tissue_stages
        )
        tissue_mask, tissue_features = self.tissue_decoder(tissue_bottleneck, tissue_skips)

        # === TISSUE → NUCLEI (STOP GRADIENT) ===
        tissue_attn = self.tissue_attention_encoder(tissue_features.detach())

        # === NUCLEI STREAM ===
        nuclei_bottleneck, nuclei_skips = self._encode(
            x, self.nuclei_stem, self.nuclei_down, self.nuclei_stages
        )

        B, C, H, W = nuclei_bottleneck.shape
        nq = nuclei_bottleneck.flatten(2).transpose(1, 2)
        nt = tissue_attn.flatten(2).transpose(1, 2)
        fused = self.cross_attention(nq, context=nt)
        fused = fused.transpose(1, 2).reshape(B, C, H, W)

        nuclei_mask, _ = self.nuclei_decoder(fused, nuclei_skips)

        return tissue_mask, nuclei_mask
