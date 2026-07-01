"""Research-only modules excluded from the production baseline."""

from .attention import LocalGlobalAttention
from .moe import Expert, SparseMoE
from .transformer import EncoderTransformerBlock, EncoderTransformerStack

__all__ = [
    "EncoderTransformerBlock",
    "EncoderTransformerStack",
    "Expert",
    "LocalGlobalAttention",
    "SparseMoE",
]
