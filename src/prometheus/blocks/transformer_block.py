from __future__ import annotations

import torch
import torch.nn as nn

from .attention import LocalGlobalAttention
from .moe import TopKPagedMoE


class SparseMoELocalGlobalEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int,
        num_experts: int = 8, top_k: int = 2, window_size: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = LocalGlobalAttention(d_model, n_heads, window_size)
        self.moe = TopKPagedMoE(d_model, d_ff, num_experts, top_k)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)

        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)

        return x
