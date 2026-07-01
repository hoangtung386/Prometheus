import torch

from prometheus.legacy import UNetTissue
from prometheus.legacy.config import ModelConfig


def test_unet_tissue_default() -> None:
    model = UNetTissue()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 6, 256, 256)


def test_unet_tissue_multi_class() -> None:
    cfg = ModelConfig(in_chans=4, num_classes=5)
    model = UNetTissue(config=cfg)
    x = torch.randn(2, 4, 128, 128)
    out = model(x)
    assert out.shape == (2, 5, 128, 128)


def test_unet_tissue_different_sizes() -> None:
    model = UNetTissue()
    for size in [128, 256, 512]:
        x = torch.randn(1, 3, size, size)
        out = model(x)
        assert out.shape == (1, 6, size, size), f"Failed at size {size}"


def test_unet_tissue_preserves_non_stride_aligned_size() -> None:
    cfg = ModelConfig(encoder_dims=[16, 32, 64, 128], encoder_depths=[1, 1, 1, 1])
    model = UNetTissue(cfg)
    out = model(torch.randn(1, 3, 130, 158))
    assert out.shape == (1, 6, 130, 158)


def test_unet_tissue_gradient_flow() -> None:
    model = UNetTissue()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
