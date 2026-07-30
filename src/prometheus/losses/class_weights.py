"""Class weights for the heavily imbalanced PUMA tasks.

A full inverse-frequency weighting is the wrong default here, for two reasons observed on
this dataset:

1. **It rewards the unscored class.** ``tissue_white_background`` is the *rarest* tissue
   label (~0.5% of pixels), so plain inverse frequency hands it the largest weight — 3.78
   against 1.05 for necrosis in a real run — while the challenge does not score it at all.
   The cross-entropy gradient then mostly pushes the model towards predicting background,
   and every background pixel it wins where a scored class belongs is a false negative.
2. **The dynamic range is unusable.** Inverse frequency spanned 420:1 between background
   and tumour, which destabilised early training (first-epoch loss above 40).

:func:`compute_class_weights` therefore defaults to a square-root inverse frequency
normalised over the *scored* classes only, with an explicit small weight for background.
See ``docs/phan-tich-tissue-va-ke-hoach.md`` section 3.3.
"""

from __future__ import annotations

import torch

__all__ = [
    "BACKGROUND_TISSUE_WEIGHT",
    "class_weights_from_counts",
    "compute_class_weights",
]

BACKGROUND_TISSUE_WEIGHT = 0.1
"""Fixed weight for the unscored ``background`` tissue channel.

Non-zero so the model still learns to place background, small so it cannot outvote the
five classes the leaderboard actually measures.
"""


def class_weights_from_counts(
    counts: torch.Tensor,
    power: float = 0.5,
    reserved_prefix: int = 0,
    reserved_weight: float = BACKGROUND_TISSUE_WEIGHT,
) -> torch.Tensor:
    """Return ``counts ** -power`` weights, normalised to mean 1 over the scored classes.

    Args:
        counts: Per-class support (pixels for tissue, instances for nuclei).
        power: Exponent on the inverse frequency. ``0.5`` (the default) is square-root
            inverse frequency; ``1.0`` reproduces plain inverse frequency.
        reserved_prefix: Number of leading channels excluded from normalisation and
            assigned ``reserved_weight`` instead. Use ``1`` for tissue (background) and
            ``0`` for nuclei, whose channels are all scored.
        reserved_weight: Weight assigned to the reserved leading channels.

    Classes with zero support get weight 1.0; they never contribute to the loss, so the
    value only needs to keep the tensor finite.
    """
    if not 0.0 < power <= 1.0:
        raise ValueError("power must be in (0, 1]")
    if reserved_prefix < 0 or reserved_prefix >= len(counts):
        raise ValueError(f"reserved_prefix must be in [0, {len(counts)})")

    weights = torch.ones(len(counts), dtype=torch.float32)
    scored = counts[reserved_prefix:].float()
    present = scored > 0
    if present.any():
        inverse = scored[present].pow(-power)
        scored_weights = torch.ones(len(scored), dtype=torch.float32)
        scored_weights[present] = inverse / inverse.mean()
        weights[reserved_prefix:] = scored_weights
    weights[:reserved_prefix] = reserved_weight
    return weights


@torch.no_grad()
def compute_class_weights(
    loader,
    num_tissue_classes: int,
    num_nucleus_types: int,
    power: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan ``loader`` once and return ``(tissue_weights, nuclei_weights)`` on the CPU.

    Tissue weights come from per-class pixel counts and nuclei weights from per-class
    instance counts. Augmentation does not change either histogram, so iterating the train
    loader is safe. The caller is expected to cache the result: this is a full data pass.
    """
    tissue_counts = torch.zeros(num_tissue_classes, dtype=torch.float64)
    nuclei_counts = torch.zeros(num_nucleus_types, dtype=torch.float64)
    for batch in loader:
        mask = batch.tissue.mask.detach().cpu().reshape(-1)
        tissue_counts += torch.bincount(mask, minlength=num_tissue_classes)[:num_tissue_classes]
        for target in batch.nuclei:
            if target.labels.numel():
                labels = target.labels.detach().cpu().reshape(-1)
                nuclei_counts += torch.bincount(labels, minlength=num_nucleus_types)[:num_nucleus_types]
    return (
        class_weights_from_counts(tissue_counts, power=power, reserved_prefix=1),
        class_weights_from_counts(nuclei_counts, power=power, reserved_prefix=0),
    )
