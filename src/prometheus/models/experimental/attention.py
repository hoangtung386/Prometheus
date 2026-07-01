from __future__ import annotations

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

    def _window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        B, heads, L, d_head = q.shape
        height, width = spatial_size
        if height * width != L:
            raise ValueError(f"spatial_size {spatial_size} does not match sequence length {L}")
        window = min(self.window_size, height, width)
        pad_h = (-height) % window
        pad_w = (-width) % window

        def partition(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(B, heads, height, width, d_head).permute(0, 1, 4, 2, 3)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
            padded_h, padded_w = height + pad_h, width + pad_w
            tensor = tensor.reshape(B, heads, d_head, padded_h // window, window, padded_w // window, window)
            return tensor.permute(0, 1, 3, 5, 4, 6, 2).reshape(-1, window * window, d_head)

        q_windows, k_windows, v_windows = partition(q), partition(k), partition(v)
        attention = (q_windows @ k_windows.transpose(-2, -1)) * (d_head**-0.5)
        output = F.softmax(attention, dim=-1) @ v_windows
        padded_h, padded_w = height + pad_h, width + pad_w
        output = output.reshape(B, heads, padded_h // window, padded_w // window, window, window, d_head)
        output = output.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, heads, d_head, padded_h, padded_w)
        return output[:, :, :, :height, :width].permute(0, 1, 3, 4, 2).reshape(B, heads, L, d_head)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        spatial_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
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

        attn_glob = (q_glob @ k_glob.transpose(-2, -1)) * (self.d_head**-0.5)
        attn_glob = F.softmax(attn_glob, dim=-1)
        out_glob = attn_glob @ v_glob

        S = context.shape[1] if context is not None else L
        if S != L:
            raise ValueError("Local cross-attention requires aligned query and context spatial sizes")
        if spatial_size is None:
            side = int(L**0.5)
            spatial_size = (side, side) if side * side == L else (1, L)
        out_loc = self._window_attention(q_loc, k_loc, v_loc, spatial_size)

        out = torch.cat([out_loc, out_glob], dim=1)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
