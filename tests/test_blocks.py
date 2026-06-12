import torch
import pytest

from prometheus.blocks import (
    ConvNeXtBlock,
    DecoderBlock,
    EncoderTransformerBlock,
    EncoderTransformerStack,
    LocalGlobalAttention,
)
from prometheus.models._base_unet import forward_decoder


def test_convnext_block() -> None:
    block = ConvNeXtBlock(dim=64, drop_path=0.1)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_convnext_block_no_drop_path() -> None:
    block = ConvNeXtBlock(dim=64)
    x = torch.randn(2, 64, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 16, 16)


def test_decoder_block_upsample() -> None:
    block = DecoderBlock(dim=64, has_upsample=True, in_dim=128)
    x = torch.randn(2, 128, 16, 16)
    skip = torch.randn(2, 64, 32, 32)
    out = block(x, skip)
    assert out.shape == (2, 64, 32, 32)


def test_decoder_block_no_upsample() -> None:
    block = DecoderBlock(dim=64)
    x = torch.randn(2, 64, 32, 32)
    skip = torch.randn(2, 64, 32, 32)
    out = block(x, skip)
    assert out.shape == (2, 64, 32, 32)


def test_local_global_attention() -> None:
    attn = LocalGlobalAttention(d_model=256, n_heads=8, window_size=14)
    x = torch.randn(2, 196, 256)
    out = attn(x)
    assert out.shape == (2, 196, 256)


def test_decoder_block_interpolation() -> None:
    block = DecoderBlock(dim=64, has_upsample=True, in_dim=128)
    x = torch.randn(2, 128, 16, 16)
    skip = torch.randn(2, 64, 28, 28)
    out = block(x, skip)
    assert out.shape == (2, 64, 28, 28)


def test_forward_decoder_rejects_skip_depth_mismatch() -> None:
    levels = torch.nn.ModuleList([
        torch.nn.ModuleList([DecoderBlock(dim=64, has_upsample=True, in_dim=128)]),
        torch.nn.ModuleList([DecoderBlock(dim=32, has_upsample=True, in_dim=64)]),
        torch.nn.ModuleList([DecoderBlock(dim=16, has_upsample=True, in_dim=32)]),
    ])
    output_head = torch.nn.Identity()
    x = torch.randn(1, 128, 8, 8)
    skips = [
        [torch.randn(1, 16, 64, 64)],
        [torch.randn(1, 32, 32, 32)],
        [torch.randn(1, 64, 16, 16), torch.randn(1, 64, 16, 16)],
    ]
    with pytest.raises(ValueError, match="expected 1 skip tensors"):
        forward_decoder(x, skips, levels, output_head)


def test_encoder_transformer_block_self_attn_only() -> None:
    block = EncoderTransformerBlock(
        d_model=128, n_heads=4, d_ff=512,
        d_expert=32, num_experts=4, top_k=2, window_size=8,
    )
    x = torch.randn(2, 64, 128)
    out, moe_loss = block(x)
    assert out.shape == (2, 64, 128)
    assert moe_loss.item() >= 0


def test_encoder_transformer_block_cross_attn() -> None:
    block = EncoderTransformerBlock(
        d_model=128, n_heads=4, d_ff=512,
        d_expert=32, num_experts=4, top_k=2, window_size=8,
    )
    x = torch.randn(2, 64, 128)
    context = torch.randn(2, 64, 128)
    out, _ = block(x, context=context)
    assert out.shape == (2, 64, 128)


def test_encoder_transformer_stack() -> None:
    stack = EncoderTransformerStack(
        num_blocks=3,
        d_model=64, n_heads=4, d_ff=256,
        d_expert=16, num_experts=4, top_k=2, window_size=8,
    )
    x = torch.randn(2, 64, 64)
    context = torch.randn(2, 64, 64)
    out, total_loss = stack(x, context=context)
    assert out.shape == (2, 64, 64)
    assert total_loss.item() >= 0


def test_transformer_gradient_flow() -> None:
    block = EncoderTransformerBlock(
        d_model=64, n_heads=4, d_ff=256,
        d_expert=16, num_experts=4, top_k=2, window_size=8,
    )
    x = torch.randn(2, 16, 64, requires_grad=True)
    out, loss = block(x)
    (out.sum() + loss).backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
