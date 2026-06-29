"""Canonical neural building blocks — re-exports from blocks for the new target layout."""

from ..blocks import (  # noqa: F401
    ConvNeXtBlock,
    DecoderBlock,
    EncoderTransformerBlock,
    EncoderTransformerStack,
    Expert,
    LocalGlobalAttention,
    SparseMoE,
)

__all__ = [
    "ConvNeXtBlock",
    "DecoderBlock",
    "EncoderTransformerBlock",
    "EncoderTransformerStack",
    "Expert",
    "LocalGlobalAttention",
    "SparseMoE",
]
