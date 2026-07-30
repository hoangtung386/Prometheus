"""Prometheus: tissue segmentation and nuclei detection for the PUMA challenge.

Import from :mod:`prometheus.api` to compose a run; this module only re-exports the model
and its output contract so ``import prometheus`` stays cheap and free of optional
dependencies.
"""

from .models import MultitaskOutput, PrometheusNet

__version__ = "0.5.0"

__all__ = ["MultitaskOutput", "PrometheusNet", "__version__"]
