import torch

from prometheus.blocks import (
    ConvNeXtBlock,
    DecoderBlock,
    LocalGlobalAttention,
    TopKPagedMoE,
    SparseMoELocalGlobalEncoderLayer,
)


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


def test_topk_moe() -> None:
    moe = TopKPagedMoE(d_model=256, d_ff=1024, num_experts=8, top_k=2)
    x = torch.randn(2, 49, 256)
    out = moe(x)
    assert out.shape == (2, 49, 256)


def test_sparse_moe_transformer_layer() -> None:
    layer = SparseMoELocalGlobalEncoderLayer(
        d_model=256, n_heads=8, d_ff=1024, num_experts=8, window_size=16
    )
    x = torch.randn(2, 256, 256)
    out = layer(x)
    assert out.shape == (2, 256, 256)


def test_decoder_block_interpolation() -> None:
    block = DecoderBlock(dim=64, has_upsample=True, in_dim=128)
    x = torch.randn(2, 128, 16, 16)
    skip = torch.randn(2, 64, 28, 28)  # non-matching size
    out = block(x, skip)
    assert out.shape == (2, 64, 28, 28)
