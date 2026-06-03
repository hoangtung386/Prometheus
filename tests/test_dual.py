import torch

from prometheus import DualUNet
from prometheus.config import ModelConfig


def test_dual_unet_forward() -> None:
    model = DualUNet()
    x = torch.randn(2, 3, 256, 256)
    tissue_mask, nuclei_mask, moe_loss = model(x)
    assert tissue_mask.shape == (2, 1, 256, 256)
    assert nuclei_mask.shape == (2, 1, 256, 256)
    assert moe_loss.item() >= 0


def test_dual_unet_multi_class() -> None:
    cfg = ModelConfig(in_chans=3, num_classes=3)
    model = DualUNet(config=cfg)
    x = torch.randn(1, 3, 128, 128)
    t, n, _ = model(x)
    assert t.shape == (1, 3, 128, 128)
    assert n.shape == (1, 3, 128, 128)


def test_dual_unet_different_sizes() -> None:
    model = DualUNet()
    for size in [64, 128, 256]:
        x = torch.randn(1, 3, size, size)
        t, n, _ = model(x)
        assert t.shape == (1, 1, size, size), f"tissue failed at {size}"
        assert n.shape == (1, 1, size, size), f"nuclei failed at {size}"


def test_dual_unet_gradient_flow() -> None:
    model = DualUNet()
    x = torch.randn(2, 3, 128, 128)
    t, n, moe_loss = model(x)
    loss = t.sum() + n.sum() + moe_loss
    loss.backward()
    named_params = list(model.named_parameters())
    assert len(named_params) > 0
    has_any_grad = False
    for name, param in named_params:
        if param.grad is not None:
            has_any_grad = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    assert has_any_grad, "No gradients computed at all"


def test_dual_unet_stop_gradient() -> None:
    cfg = ModelConfig(in_chans=3, num_classes=1)
    model = DualUNet(config=cfg)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    _, nuclei_mask, moe_loss = model(x)
    loss = nuclei_mask.sum() + moe_loss
    loss.backward()
    tissue_decoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.tissue_decoder.named_parameters()
    )
    assert not tissue_decoder_grad, "Tissue decoder should NOT get grad from nuclei loss (stop_gradient)"

    model.zero_grad()
    t, n, ml = model(x)
    (t.sum() + n.sum() + ml).backward()
    checks = [
        ("tissue_stem", model.tissue_stem),
        ("tissue_stages", model.tissue_stages),
        ("tissue_decoder", model.tissue_decoder),
        ("tissue_attention_encoder", model.tissue_attention_encoder),
        ("transformer", model.transformer),
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
    t, n, _ = model(x)
    assert t.shape == (1, 1, 256, 256)
    assert n.shape == (1, 1, 256, 256)


def test_dual_unet_transformer_context() -> None:
    cfg = ModelConfig(
        in_chans=3, num_classes=1,
        num_transformer_blocks=2,
        num_experts=8, moe_top_k=2,
    )
    model = DualUNet(config=cfg)
    x = torch.randn(2, 3, 128, 128)
    t, n, ml = model(x)
    assert t.shape == (2, 1, 128, 128)
    assert n.shape == (2, 1, 128, 128)
