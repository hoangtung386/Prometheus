from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from ._base_unet import (
    build_decoder,
    build_encoder,
    forward_decoder,
    forward_encoder,
)


class Encoder(nn.Module):
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


class Decoder(nn.Module):
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
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        return self.decoder(x, skips)
