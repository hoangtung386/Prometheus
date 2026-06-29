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
        tokens_per_expert = torch.zeros(self.num_experts, device=gate_weights.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = (topk_indices == i).any(dim=-1).float().sum()
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
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            matched = topk_indices[idx] == i
            k_idx = matched.int().argmax(dim=-1)
            w = topk_weights[idx, k_idx].unsqueeze(-1)
            out_down[idx] += w * expert(down[idx])

        aux_loss = self._load_balancing_loss(gate_weights, topk_indices)

        out = self.up_proj(out_down)
        return out.reshape(B, L, D), aux_loss
