from __future__ import annotations

import torch
import torch.nn as nn

from .attention import LocalGlobalAttention
from .moe import SparseMoE


class EncoderTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_expert: int = 256,
        num_experts: int = 512,
        top_k: int = 8,
        window_size: int = 4,
        dropout: float = 0.1,
        use_moe: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = LocalGlobalAttention(d_model, n_heads, window_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.cross_attn = LocalGlobalAttention(d_model, n_heads, window_size)
        self.moe = (
            SparseMoE(
                d_model=d_model,
                d_expert=d_expert,
                num_experts=num_experts,
                top_k=top_k,
            )
            if use_moe
            else None
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)
        self.norm7 = nn.LayerNorm(d_model)
        self.norm8 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        spatial_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.norm2(self.self_attn(self.norm1(x), spatial_size=spatial_size)))
        x = x + self.dropout(self.norm4(self.ffn(self.norm3(x))))
        if context is not None:
            x = x + self.dropout(
                self.norm6(self.cross_attn(self.norm5(x), context=context, spatial_size=spatial_size))
            )
        if self.moe is not None:
            moe_out, moe_loss = self.moe(self.norm7(x))
            x = x + self.dropout(self.norm8(moe_out))
        else:
            moe_loss = x.new_zeros(())
        return x, moe_loss


class EncoderTransformerStack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_expert: int = 256,
        num_experts: int = 512,
        top_k: int = 8,
        window_size: int = 4,
        dropout: float = 0.1,
        use_moe: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    d_expert,
                    num_experts,
                    top_k,
                    window_size,
                    dropout,
                    use_moe,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        spatial_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_loss = 0.0
        for block in self.blocks:
            x, loss = block(x, context=context, spatial_size=spatial_size)
            total_loss = total_loss + loss
        return x, total_loss
