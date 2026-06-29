from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks import ConvNeXtBlock, DecoderBlock
from ..utils import LayerNorm


def build_encoder(
    in_chans: int,
    dims: list[int],
    depths: list[int],
    drop_path_rate: float = 0.0,
    block_cls: type = ConvNeXtBlock,
) -> tuple[nn.Module, nn.ModuleList, nn.ModuleList]:
    stem = nn.Sequential(
        nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
    )

    downsample_layers = nn.ModuleList()
    for i in range(3):
        downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
        )

    stages = nn.ModuleList()
    dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    cur = 0
    for i in range(4):
        stage = nn.Sequential(*[block_cls(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
        stages.append(stage)
        cur += depths[i]

    return stem, downsample_layers, stages


def forward_encoder(
    x: torch.Tensor,
    stem: nn.ModuleList,
    downsample_layers: nn.ModuleList,
    stages: nn.ModuleList,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    all_skips: list[list[torch.Tensor]] = []

    x = stem(x)
    stage_skips: list[torch.Tensor] = []
    for layer in stages[0]:
        x = layer(x)
        stage_skips.append(x)
    all_skips.append(stage_skips)

    for i in range(3):
        x = downsample_layers[i](x)
        stage_skips = []
        for layer in stages[i + 1]:
            x = layer(x)
            stage_skips.append(x)
        all_skips.append(stage_skips)

    return x, all_skips[:-1]


def build_decoder(
    encoder_dims: list[int],
    encoder_depths: list[int],
    num_classes: int = 1,
    block_cls: type[nn.Module] = DecoderBlock,
) -> tuple[nn.ModuleList, nn.Module]:
    if len(encoder_dims) != len(encoder_depths):
        raise ValueError(
            f"encoder_dims and encoder_depths must have the same length, got "
            f"{len(encoder_dims)} and {len(encoder_depths)}"
        )
    if len(encoder_dims) < 4:
        raise ValueError("decoder expects at least 4 encoder stages")

    rev_dims = encoder_dims[::-1]
    dec_depths = list(reversed(encoder_depths[:-1]))

    levels = nn.ModuleList()
    for level in range(3):
        dim = rev_dims[level + 1]
        in_dim = rev_dims[level]
        depth = dec_depths[level]

        layers = nn.ModuleList()
        for j in range(depth):
            has_upsample = j == 0
            layers.append(
                block_cls(
                    dim=dim,
                    has_upsample=has_upsample,
                    in_dim=in_dim if has_upsample else None,
                )
            )
        levels.append(layers)

    output_head = nn.Sequential(
        nn.ConvTranspose2d(encoder_dims[0], encoder_dims[0], kernel_size=4, stride=4),
        nn.Conv2d(encoder_dims[0], num_classes, kernel_size=1),
    )

    return levels, output_head


def forward_decoder(
    x: torch.Tensor,
    skips: list[list[torch.Tensor]],
    levels: nn.ModuleList,
    output_head: nn.Module,
) -> torch.Tensor:
    for level in range(3):
        stage_skips = skips[2 - level]
        if len(stage_skips) != len(levels[level]):
            raise ValueError(
                f"Decoder level {level} expected {len(levels[level])} skip tensors, got {len(stage_skips)}"
            )
        for j, layer in enumerate(levels[level]):
            x = layer(x, stage_skips[j])
    x = output_head(x)
    return x
