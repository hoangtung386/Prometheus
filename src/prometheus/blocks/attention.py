from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        self.local_heads = n_heads // 2
        self.global_heads = n_heads - self.local_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _format_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    @staticmethod
    def _get_valid_window_size(seq_len: int, window_size: int) -> int:
        for w in range(min(window_size, seq_len), 0, -1):
            if seq_len % w == 0:
                return w
        return 1

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape

        q = self._format_heads(self.q_proj(x))

        if context is not None:
            k = self._format_heads(self.k_proj(context))
            v = self._format_heads(self.v_proj(context))
        else:
            k = self._format_heads(self.k_proj(x))
            v = self._format_heads(self.v_proj(x))

        q_loc, q_glob = torch.split(q, [self.local_heads, self.global_heads], dim=1)
        k_loc, k_glob = torch.split(k, [self.local_heads, self.global_heads], dim=1)
        v_loc, v_glob = torch.split(v, [self.local_heads, self.global_heads], dim=1)

        attn_glob = (q_glob @ k_glob.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn_glob = F.softmax(attn_glob, dim=-1)
        out_glob = attn_glob @ v_glob

        W = self._get_valid_window_size(L, self.window_size)
        S = context.shape[1] if context is not None else L
        num_windows = L // W

        q_loc_w = q_loc.view(B, self.local_heads, num_windows, W, self.d_head).reshape(-1, W, self.d_head)

        if context is not None:
            S_w = S // num_windows
            if S % num_windows != 0:
                S_w = self._get_valid_window_size(S, S // num_windows) if S // num_windows > 0 else 1
            k_loc_w = k_loc.view(B, self.local_heads, num_windows, S_w, self.d_head).reshape(-1, S_w, self.d_head)
            v_loc_w = v_loc.view(B, self.local_heads, num_windows, S_w, self.d_head).reshape(-1, S_w, self.d_head)
        else:
            k_loc_w = k_loc.view(B, self.local_heads, num_windows, W, self.d_head).reshape(-1, W, self.d_head)
            v_loc_w = v_loc.view(B, self.local_heads, num_windows, W, self.d_head).reshape(-1, W, self.d_head)

        attn_loc = (q_loc_w @ k_loc_w.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn_loc = F.softmax(attn_loc, dim=-1)
        out_loc_w = attn_loc @ v_loc_w

        out_loc = out_loc_w.view(
            B, self.local_heads, num_windows, W, self.d_head
        ).reshape(B, self.local_heads, L, self.d_head)

        out = torch.cat([out_loc, out_glob], dim=1)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
