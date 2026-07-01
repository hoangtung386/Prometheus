"""Stable low-level neural network blocks used by the production model.

Experimental modules (attention, MoE, transformer) live under
:mod:`prometheus.models.experimental` and are excluded from the baseline
import path to prevent accidental use.
"""

from .convnext_block import ConvNeXtBlock
from .decoder_block import DecoderBlock

__all__ = [
    "ConvNeXtBlock",
    "DecoderBlock",
]
