"""Modern inference API — use :mod:`prometheus.legacy` for semantic postprocessing."""

from .nuclei_decoder import decode_nuclei
from .predictor import MultitaskPrediction, PrometheusPredictor

__all__ = [
    "MultitaskPrediction",
    "PrometheusPredictor",
    "decode_nuclei",
]
