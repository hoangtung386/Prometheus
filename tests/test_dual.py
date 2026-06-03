import torch

from prometheus import DualUNet
from prometheus.blocks import CrossAttention
from prometheus.config import ModelConfig


def test_dual_unet_forward() -> None:
    model = DualUNet()
    x = torch.randn(2, 3, 256, 256)
    tissue_mask, nuclei_mask = model(x)
    assert tissue_mask.shape == (2, 1, 256, 256)
    assert nuclei_mask.shape == (2, 1, 256, 256)


def test_dual_unet_multi_class() -> None:
    cfg = ModelConfig(in_chans=3, num_classes=3)
    model = DualUNet(config=cfg)
    x = torch.randn(1, 3, 128, 128)
    t, n = model(x)
    assert t.shape == (1, 3, 128, 128)
    assert n.shape == (1, 3, 128, 128)


def test_dual_unet_different_sizes() -> None:
    model = DualUNet()
    for size in [64, 128, 256]:
        x = torch.randn(1, 3, size, size)
        t, n = model(x)
        assert t.shape == (1, 1, size, size), f"tissue failed at {size}"
        assert n.shape == (1, 1, size, size), f"nuclei failed at {size}"


def test_dual_unet_gradient_flow() -> None:
    model = DualUNet()
    x = torch.randn(2, 3, 128, 128)
    t, n = model(x)
    loss = t.sum() + n.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_dual_unet_stop_gradient() -> None:
    """Verify stop-gradient prevents nuclei gradient from flowing into tissue decoder."""
    cfg = ModelConfig(in_chans=3, num_classes=1)
    model = DualUNet(config=cfg)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    _, nuclei_mask = model(x)
    loss = nuclei_mask.sum()
    loss.backward()
    tissue_decoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.tissue_decoder.named_parameters()
    )
    assert not tissue_decoder_grad, "Tissue decoder should NOT get grad from nuclei loss (stop_gradient)"

    # Both masks together produce gradients in all major components
    model.zero_grad()
    t, n = model(x)
    (t.sum() + n.sum()).backward()
    # Check representative parameters from each component
    checks = [
        ("tissue_stem", model.tissue_stem),
        ("tissue_stages", model.tissue_stages),
        ("tissue_decoder", model.tissue_decoder),
        ("tissue_attention_encoder", model.tissue_attention_encoder),
        ("cross_attention", model.cross_attention),
        ("nuclei_stem", model.nuclei_stem),
        ("nuclei_stages", model.nuclei_stages),
        ("nuclei_decoder", model.nuclei_decoder),
    ]
    for name, module in checks:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
        assert has_grad, f"No gradient found in {name}"


def test_dual_unet_default_config() -> None:
    model = DualUNet()
    x = torch.randn(1, 3, 256, 256)
    t, n = model(x)
    assert t.shape == (1, 1, 256, 256)
    assert n.shape == (1, 1, 256, 256)


def test_cross_attention() -> None:
    attn = CrossAttention(d_model=128, n_heads=4)
    q = torch.randn(2, 16, 128)
    kv = torch.randn(2, 16, 128)
    out = attn(q, kv)
    assert out.shape == (2, 16, 128)


def test_cross_attention_different_lengths() -> None:
    attn = CrossAttention(d_model=64, n_heads=4)
    q = torch.randn(2, 16, 64)
    kv = torch.randn(2, 32, 64)
    out = attn(q, kv)
    assert out.shape == (2, 16, 64)
