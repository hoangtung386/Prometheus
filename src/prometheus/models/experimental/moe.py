from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_expert: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_expert, d_ff)
        self.fc2 = nn.Linear(d_ff, d_expert)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_expert: int = 256,
        num_experts: int = 512,
        top_k: int = 8,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.down_proj = nn.Linear(d_model, d_expert)
        self.up_proj = nn.Linear(d_expert, d_model)
        self.gate = nn.Linear(d_model, num_experts)

        self.experts = nn.ModuleList([Expert(d_expert, d_expert * 4) for _ in range(num_experts)])

    def _load_balancing_loss(
        self,
        gate_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        N = gate_weights.shape[0]
        tokens_per_expert = torch.bincount(topk_indices.reshape(-1), minlength=self.num_experts).float()
        fraction = tokens_per_expert / (N * self.top_k)
        prob = gate_weights.mean(dim=0)
        loss = self.num_experts * (fraction * prob).sum()
        return loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        tokens = x.reshape(-1, D)

        down = self.down_proj(tokens)

        gate_logits = self.gate(tokens)
        gate_weights = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out_down = torch.zeros_like(down)
        capacity = max(1, int(self.capacity_factor * tokens.shape[0] * self.top_k / self.num_experts))
        for i, expert in enumerate(self.experts):
            token_idx, route_idx = (topk_indices == i).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            route_weights = topk_weights[token_idx, route_idx]
            if token_idx.numel() > capacity:
                keep = route_weights.topk(capacity).indices
                token_idx = token_idx[keep]
                route_weights = route_weights[keep]
            contribution = route_weights.unsqueeze(-1) * expert(down[token_idx])
            out_down.index_add_(0, token_idx, contribution)

        aux_loss = self._load_balancing_loss(gate_weights, topk_indices)

        out = self.up_proj(out_down)
        return out.reshape(B, L, D), aux_loss
