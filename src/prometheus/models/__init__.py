"""Production model API — legacy architectures are registered lazily."""

from .contracts import FeaturePyramid, MultitaskOutput
from .multitask.prometheus_net import PrometheusNet
from .registry import create_model, register_model, registered_models


def _legacy_dual_factory(config):
    from ..legacy import DualUNet

    return DualUNet(config)


def _legacy_tissue_factory(config):
    from ..legacy import UNetTissue

    return UNetTissue(config)


register_model("prometheus_multitask_v1", lambda config: PrometheusNet(config))
register_model("legacy_dual_unet", _legacy_dual_factory)
register_model("tissue_convnext_unet", _legacy_tissue_factory)


__all__ = [
    "FeaturePyramid",
    "MultitaskOutput",
    "PrometheusNet",
    "create_model",
    "register_model",
    "registered_models",
]
