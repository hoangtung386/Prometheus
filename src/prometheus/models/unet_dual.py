from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..blocks.transformer_block import EncoderTransformerStack
from ..config import ModelConfig
from ..utils import LayerNorm
from ._base_unet import build_decoder, build_encoder, forward_decoder, forward_encoder


class TissueAttentionEncoder(nn.Module):
    """Encoder_Attention: encodes tissue decoder features into bottleneck context.

    Matches the diagram: Output_Tissue → Stop Gradient → Encoder_Attention
    """

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
    def __init__(self, encoder_dims: List[int], encoder_depths: List[int], num_classes: int = 1) -> None:
        super().__init__()
        self.levels, output_head = build_decoder(encoder_dims, encoder_depths, num_classes)
        self.feature_head = output_head[0]
        self.mask_head = output_head[1]

    def forward(
        self, x: torch.Tensor, skips: List[List[torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for level in range(3):
            stage_skips = skips[2 - level]
            for j, layer in enumerate(self.levels[level]):
                x = layer(x, stage_skips[j])
        full_res_feat = self.feature_head(x)
        mask = self.mask_head(full_res_feat)
        return mask, full_res_feat


class DualUNet(nn.Module):
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
            num_classes=config.num_tissue_classes,
        )

        # === TISSUE → NUCLEI BRIDGE (STOP GRADIENT) ===
        self.tissue_attention_encoder = TissueAttentionEncoder(
            in_channels=dims[0],
            dims=dims,
        )

        # === NUCLEI STREAM ===
        self.nuclei_stem, self.nuclei_down, self.nuclei_stages = build_encoder(
            in_chans=config.in_chans,
            dims=dims,
            depths=depths,
            drop_path_rate=config.drop_path_rate,
        )
        self.transformer = EncoderTransformerStack(
            num_blocks=config.num_transformer_blocks,
            d_model=dims[-1],
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            d_expert=config.d_expert,
            num_experts=config.num_experts,
            top_k=config.moe_top_k,
            window_size=config.window_size,
        )
        self.nuclei_decoder, self.nuclei_head = build_decoder(
            encoder_dims=dims,
            encoder_depths=depths,
            num_classes=config.num_nuclei_classes,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # === TISSUE STREAM ===
        t_bottleneck, t_skips = forward_encoder(
            x, self.tissue_stem, self.tissue_down, self.tissue_stages,
        )
        t_mask, t_full_res_feat = self.tissue_decoder(t_bottleneck, t_skips)

        # === TISSUE → NUCLEI CONTEXT (STOP GRADIENT) ===
        t_context = self.tissue_attention_encoder(t_full_res_feat.detach())
        B, C, H, W = t_context.shape
        context_seq = t_context.flatten(2).transpose(1, 2)

        # === NUCLEI STREAM ===
        n_bottleneck, n_skips = forward_encoder(
            x, self.nuclei_stem, self.nuclei_down, self.nuclei_stages,
        )

        n_seq = n_bottleneck.flatten(2).transpose(1, 2)
        n_seq, moe_loss = self.transformer(n_seq, context=context_seq)

        _, L, D = n_seq.shape
        Hf = Wf = int(L ** 0.5)
        n_transformed = n_seq.transpose(1, 2).reshape(B, D, Hf, Wf)

        n_mask = forward_decoder(n_transformed, n_skips, self.nuclei_decoder, self.nuclei_head)

        return t_mask, n_mask, moe_loss
