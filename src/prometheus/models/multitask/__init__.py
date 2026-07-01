"""Multitask models — legacy :class:`DualUNet` is only loaded on explicit request."""

from .prometheus_net import PrometheusNet

__all__ = ["PrometheusNet"]
