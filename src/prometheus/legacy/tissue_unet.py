from __future__ import annotations

import torch
import torch.nn as nn

from .base_unet import (
    build_decoder,
    build_encoder,
    forward_decoder,
    forward_encoder,
)
from .config import ModelConfig


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.stem, self.downsample_layers, self.stages = build_encoder(
            in_chans=config.in_chans,
            dims=config.encoder_dims,
            depths=config.encoder_depths,
            drop_path_rate=config.drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        return forward_encoder(x, self.stem, self.downsample_layers, self.stages)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.levels, self.output_head = build_decoder(
            encoder_dims=config.encoder_dims,
            encoder_depths=config.encoder_depths,
            num_classes=config.num_classes if config.num_classes is not None else config.num_tissue_classes,
        )

    def forward(self, x: torch.Tensor, skips: list[list[torch.Tensor]]) -> torch.Tensor:
        return forward_decoder(x, skips, self.levels, self.output_head)


class UNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        x, skips = self.encoder(x)
        logits = self.decoder(x, skips)
        if logits.shape[-2:] != output_size:
            logits = torch.nn.functional.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
