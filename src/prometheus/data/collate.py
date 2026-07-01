"""Task-aware collators."""

from __future__ import annotations

import torch

from ..domain import MultitaskBatch, MultitaskSample, TissueTarget


def collate_multitask(samples: list[MultitaskSample]) -> MultitaskBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return MultitaskBatch(
        images=torch.stack([sample.image for sample in samples]),
        tissue=TissueTarget(torch.stack([sample.tissue.mask for sample in samples])),
        nuclei=[sample.nuclei for sample in samples],
        metadata=[sample.metadata for sample in samples],
    )
