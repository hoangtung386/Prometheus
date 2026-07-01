import torch

from prometheus.models.experimental import Expert, SparseMoE


def test_expert_forward() -> None:
    expert = Expert(d_expert=64, d_ff=256)
    x = torch.randn(16, 64)
    out = expert(x)
    assert out.shape == (16, 64)
    assert not torch.isnan(out).any()


def test_moe_forward() -> None:
    moe = SparseMoE(d_model=64, d_expert=32, num_experts=8, top_k=2)
    x = torch.randn(2, 16, 64)
    out, aux_loss = moe(x)
    assert out.shape == (2, 16, 64)
    assert out.shape[-1] == 64
    assert aux_loss.item() >= 0


def test_moe_down_up_projection() -> None:
    d_model = 128
    d_expert = 32
    moe = SparseMoE(d_model=d_model, d_expert=d_expert, num_experts=16, top_k=2)
    x = torch.randn(2, 16, d_model)
    out, _ = moe(x)
    assert out.shape == (2, 16, d_model)


def test_moe_large_experts() -> None:
    moe = SparseMoE(d_model=128, d_expert=64, num_experts=512, top_k=8)
    x = torch.randn(1, 64, 128)
    out, aux_loss = moe(x)
    assert out.shape == (1, 64, 128)
    assert aux_loss.item() >= 0


def test_moe_small_batch() -> None:
    moe = SparseMoE(d_model=32, d_expert=16, num_experts=16, top_k=2)
    x = torch.randn(1, 4, 32)
    out, aux_loss = moe(x)
    assert out.shape == (1, 4, 32)


def test_moe_gradient_flow() -> None:
    moe = SparseMoE(d_model=64, d_expert=32, num_experts=8, top_k=2)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out, aux_loss = moe(x)
    loss = out.sum() + aux_loss
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
