from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TopKPagedMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

        self.gate = nn.Linear(d_model, num_experts)
        self.noise_linear = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        tokens = x.view(-1, D)

        gate_logits = self.gate(tokens)
        if self.training:
            noise_logits = F.softplus(self.noise_linear(tokens))
            eps = torch.randn_like(gate_logits)
            gate_logits = gate_logits + (eps * noise_logits)

        gate_weights = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out_tokens = torch.zeros_like(tokens)

        for i, expert in enumerate(self.experts):
            token_indices, k_indices = (topk_indices == i).nonzero(as_tuple=True)

            if token_indices.numel() == 0:
                continue

            selected_tokens = tokens[token_indices]
            expert_outputs = expert(selected_tokens)

            weight = topk_weights[token_indices, k_indices].unsqueeze(-1)
            out_tokens[token_indices] += weight * expert_outputs

        return out_tokens.view(B, L, D)
