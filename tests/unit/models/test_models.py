import torch

from prometheus import UNetTissue
from prometheus.config import ModelConfig


def test_unet_tissue_default() -> None:
    model = UNetTissue()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


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
        assert out.shape == (1, 1, size, size), f"Failed at size {size}"


def test_unet_tissue_gradient_flow() -> None:
    model = UNetTissue()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
