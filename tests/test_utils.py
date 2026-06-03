import torch

from prometheus.utils import LayerNorm, GRN
from prometheus.utils.minkowski_utils import (
    MinkowskiGRN,
    MinkowskiDropPath,
    MinkowskiLayerNorm,
    MinkowskiEngineNotAvailableError,
)


def test_layer_norm_channels_last() -> None:
    ln = LayerNorm(64, data_format="channels_last")
    x = torch.randn(2, 16, 16, 64)
    out = ln(x)
    assert out.shape == x.shape


def test_layer_norm_channels_first() -> None:
    ln = LayerNorm(64, data_format="channels_first")
    x = torch.randn(2, 64, 16, 16)
    out = ln(x)
    assert out.shape == x.shape


def test_grn() -> None:
    grn = GRN(64)
    x = torch.randn(2, 16, 16, 64)
    out = grn(x)
    assert out.shape == x.shape


def test_minkowski_classes_not_available() -> None:
    try:
        MinkowskiGRN(dim=64)
        has_me = True
    except MinkowskiEngineNotAvailableError:
        has_me = False

    if not has_me:
        try:
            MinkowskiDropPath()
            assert False, "Should have raised"
        except MinkowskiEngineNotAvailableError:
            pass

        try:
            MinkowskiLayerNorm(64)
            assert False, "Should have raised"
        except MinkowskiEngineNotAvailableError:
            pass
