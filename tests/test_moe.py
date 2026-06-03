import torch

from prometheus.layers import MoE, HierarchicalMoE, Experts
from prometheus.layers.mixture_of_experts import HeirarchicalMoE


def test_moe_forward() -> None:
    moe = MoE(dim=64, num_experts=8)
    x = torch.randn(2, 16, 64)
    out, loss = moe(x)
    assert out.shape == (2, 16, 64)
    assert loss.item() >= 0


def test_moe_balancing_loss() -> None:
    moe = MoE(dim=64, num_experts=8, loss_coef=1e-2)
    x = torch.randn(4, 32, 64)
    _, loss = moe(x)
    assert loss.item() > 0


def test_hierarchical_moe() -> None:
    hmoe = HierarchicalMoE(dim=64, num_experts=(4, 4))
    x = torch.randn(2, 16, 64)
    out, loss = hmoe(x)
    assert out.shape == (2, 16, 64)


def test_experts() -> None:
    experts = Experts(dim=64, num_experts=8)
    x = torch.randn(8, 16, 64)
    out = experts(x)
    assert out.shape == (8, 16, 64)


def test_backward_compat_alias() -> None:
    assert HeirarchicalMoE is HierarchicalMoE
