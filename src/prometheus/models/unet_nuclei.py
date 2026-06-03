from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from ..blocks.minkowski_block import MinkowskiConvNeXtBlock
from ..config import ModelConfig
from ..utils.minkowski_utils import (
    MinkowskiConvolution,
    MinkowskiLayerNorm,
    MinkowskiLinear,
    require_minkowski_engine,
    to_sparse,
)
from ._base_unet import (
    build_decoder,
    build_encoder,
    forward_decoder,
    forward_encoder,
)


class EncoderNuclei(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.stem, self.downsample_layers, self.stages = build_encoder(
            in_chans=config.in_chans,
            dims=config.encoder_dims,
            depths=config.encoder_depths,
            drop_path_rate=config.drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[List[torch.Tensor]]]:
        return forward_encoder(x, self.stem, self.downsample_layers, self.stages)


class EncoderFeaturesMaskTissue(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        require_minkowski_engine()
        self.depths = config.encoder_depths
        dims = config.encoder_dims
        D = config.D

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(dims[-1], dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=True, dimension=D),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.encoder_depths))]
        cur = 0
        for i in range(4):
            blocks = [
                MinkowskiConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], D=D)
                for j in range(config.encoder_depths[i])
            ]
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            cur += config.encoder_depths[i]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)

    @staticmethod
    def upsample_mask(mask: torch.Tensor, scale: int) -> torch.Tensor:
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
            repeat_interleave(scale, axis=1).\
            repeat_interleave(scale, axis=2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_stages = len(self.stages)
        mask = self.upsample_mask(mask, 2 ** (num_stages - 1))
        mask = mask.unsqueeze(1).type_as(x)

        x = self.downsample_layers[0](x)
        x *= (1. - mask)

        x = to_sparse(x)

        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)

        x = x.dense()[0]
        return x


class TransformerEncoderNuclei(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.stem, self.downsample_layers, self.stages = build_encoder(
            in_chans=config.encoder_dims[-1],
            dims=config.encoder_dims,
            depths=config.encoder_depths,
            drop_path_rate=config.drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[List[torch.Tensor]]]:
        return forward_encoder(x, self.stem, self.downsample_layers, self.stages)


class DecoderNuclei(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.levels, self.output_head = build_decoder(
            encoder_dims=config.encoder_dims,
            encoder_depths=config.encoder_depths,
            num_classes=config.num_classes,
        )

    def forward(
        self, x: torch.Tensor, skips: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        return forward_decoder(x, skips, self.levels, self.output_head)


class UNet(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.encoder = EncoderNuclei(config)
        self.decoder = DecoderNuclei(config)
        self.encoder_features_mask_tissue = EncoderFeaturesMaskTissue(config)
        self.transformer_encoder_nuclei = TransformerEncoderNuclei(config)

    def forward(self, x: torch.Tensor, features_map_tissue: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        logits_tissue = self.encoder_features_mask_tissue(x, features_map_tissue)
        x = self.transformer_encoder_nuclei(x + logits_tissue)[0]
        x = self.decoder(x, skips)
        return x
