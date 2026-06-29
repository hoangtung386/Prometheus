"""Canonical normalization layers — re-exports from utils for the new target layout."""

from ..utils.norm import GRN, LayerNorm  # noqa: F401

__all__ = ["GRN", "LayerNorm"]
