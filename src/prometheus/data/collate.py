"""Collators that keep variable-length nuclei targets intact."""

from __future__ import annotations

import torch

from ..domain import MultitaskBatch, MultitaskSample, TissueTarget

__all__ = ["collate_multitask"]


def collate_multitask(samples: list[MultitaskSample]) -> MultitaskBatch:
    """Stack images and tissue masks; keep nuclei as a per-image list.

    The default collator would try to stack the nuclei tensors and fail, because instance
    counts differ per image. Padding them into a dense tensor would work but then every
    downstream consumer needs a validity mask.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return MultitaskBatch(
        images=torch.stack([sample.image for sample in samples]),
        tissue=TissueTarget(torch.stack([sample.tissue.mask for sample in samples])),
        nuclei=[sample.nuclei for sample in samples],
        metadata=[sample.metadata for sample in samples],
    )
