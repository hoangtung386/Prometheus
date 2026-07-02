"""Inverse-frequency class weights for the imbalanced PUMA tasks.

Tissue is weighted by pixel frequency and nuclei by instance frequency, both scanned
once from the training loader. Weights fight the majority-class collapse that a tiny,
imbalanced dataset otherwise produces.
"""

from __future__ import annotations

import torch


def inverse_frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    """Balanced inverse-frequency weights, normalized to mean 1 over present classes.

    Classes with zero support get weight 1.0 (they never contribute to the loss, so the
    exact value is irrelevant; 1.0 just keeps the tensor finite).
    """
    counts = counts.float()
    weights = torch.ones_like(counts)
    present = counts > 0
    if present.any():
        frequencies = counts[present] / counts[present].sum()
        inverse = 1.0 / frequencies
        weights[present] = inverse / inverse.mean()
    return weights


@torch.no_grad()
def compute_class_weights(
    loader,
    num_tissue_classes: int,
    num_nucleus_types: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan the loader once and return ``(tissue_weights, nuclei_weights)`` on the CPU.

    Tissue weights come from per-class pixel counts; nuclei weights from per-class
    instance counts. Augmentation does not change either histogram, so iterating the
    train loader is safe.
    """
    tissue_counts = torch.zeros(num_tissue_classes)
    nuclei_counts = torch.zeros(num_nucleus_types)
    for batch in loader:
        mask = batch.tissue.mask.detach().cpu().reshape(-1)
        tissue_counts += torch.bincount(mask, minlength=num_tissue_classes)[:num_tissue_classes].float()
        for target in batch.nuclei:
            if target.labels.numel():
                labels = target.labels.detach().cpu().reshape(-1)
                nuclei_counts += torch.bincount(labels, minlength=num_nucleus_types)[:num_nucleus_types].float()
    return inverse_frequency_weights(tissue_counts), inverse_frequency_weights(nuclei_counts)
