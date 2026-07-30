"""Production model API: shared backbone, task heads, fusion and typed outputs."""

from .contracts import FeaturePyramid, MultitaskOutput
from .fusion import GatedContextFusion
from .prometheus_net import PrometheusNet

__all__ = [
    "FeaturePyramid",
    "GatedContextFusion",
    "MultitaskOutput",
    "PrometheusNet",
]
