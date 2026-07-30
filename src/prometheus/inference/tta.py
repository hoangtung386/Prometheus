"""Dihedral test-time augmentation over the dense multitask maps.

The eight elements of the dihedral group (four rotations x optional flip) are all label
preserving for an H&E region of interest, so averaging over them is free accuracy at
inference cost. Two details matter:

* averaging happens in **probability space** and is mapped back to logits, so the
  downstream ``argmax`` / ``decode_nuclei`` behave exactly as they do without TTA;
* the offset map is a *signed direction*, so it must be transformed like a vector, not
  resampled like a scalar. Getting this wrong shifts every decoded centroid.

Both the validation loop and :class:`prometheus.inference.PrometheusPredictor` use this
module, so the submission path and the reported validation score agree.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..models import MultitaskOutput

__all__ = ["DIHEDRAL_VIEWS", "FLIP_VIEWS", "tta_forward"]

# (number of 90-degree rotations, horizontal flip)
DIHEDRAL_VIEWS: tuple[tuple[int, bool], ...] = (
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
)
"""All eight dihedral views. The default for inference and final validation."""

FLIP_VIEWS: tuple[tuple[int, bool], ...] = ((0, False), (0, True), (2, True))
"""Identity, horizontal flip and vertical flip. Cheaper, for per-epoch validation."""

_EPS = 1e-6


def _apply(images: torch.Tensor, rotations: int, flip: bool) -> torch.Tensor:
    transformed = torch.rot90(images, rotations, dims=(-2, -1))
    return torch.flip(transformed, dims=[-1]) if flip else transformed


def _invert(maps: torch.Tensor, rotations: int, flip: bool) -> torch.Tensor:
    restored = torch.flip(maps, dims=[-1]) if flip else maps
    return torch.rot90(restored, -rotations, dims=(-2, -1))


def _forward_offsets(offsets: torch.Tensor, rotations: int, flip: bool) -> torch.Tensor:
    """Apply a view to a two-channel ``(dx, dy)`` field.

    ``rot90(k=1)`` over ``dims=(-2, -1)`` moves the pixel at ``(x, y)`` to ``(y, W-1-x)``,
    whose differential rotates a displacement by ``(dx, dy) -> (dy, -dx)``. A horizontal
    flip negates ``dx``. Only defined here so :func:`_invert_offsets` has an executable
    specification to be tested against.
    """
    moved = _apply(offsets, rotations, flip)
    delta_x, delta_y = moved[:, 0], moved[:, 1]
    for _ in range(rotations % 4):
        delta_x, delta_y = delta_y, -delta_x
    if flip:
        delta_x = -delta_x
    return torch.stack((delta_x, delta_y), dim=1)


def _invert_offsets(offsets: torch.Tensor, rotations: int, flip: bool) -> torch.Tensor:
    """Exact inverse of :func:`_forward_offsets`.

    The view is ``rot90`` then ``flip``, so undoing it is ``flip`` then ``rot90(-k)`` — on
    the grid *and* on the vector components. Resampling the field without rotating its
    components leaves every decoded centroid displaced.
    """
    restored = _invert(offsets, rotations, flip)
    delta_x, delta_y = restored[:, 0], restored[:, 1]
    if flip:
        delta_x = -delta_x
    for _ in range(rotations % 4):
        delta_x, delta_y = -delta_y, delta_x
    return torch.stack((delta_x, delta_y), dim=1)


@torch.no_grad()
def tta_forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    views: Sequence[tuple[int, bool]] = DIHEDRAL_VIEWS,
) -> MultitaskOutput:
    """Average ``model`` over ``views`` and return logits equivalent to a single pass."""
    if not views:
        raise ValueError("At least one view is required")

    totals: dict[str, torch.Tensor] = {}
    for rotations, flip in views:
        output = model(_apply(images, rotations, flip))
        contribution = {
            "tissue": _invert(output.tissue_logits.softmax(dim=1), rotations, flip),
            "center": _invert(output.nuclei_center_logits.sigmoid(), rotations, flip),
            "classes": _invert(output.nuclei_class_logits.softmax(dim=1), rotations, flip),
            "offsets": _invert_offsets(output.nuclei_offsets, rotations, flip),
            "sizes": _invert(output.nuclei_sizes, rotations, flip),
        }
        for name, value in contribution.items():
            totals[name] = totals[name] + value if name in totals else value

    count = float(len(views))
    tissue = (totals["tissue"] / count).clamp_min(_EPS)
    center = (totals["center"] / count).clamp(_EPS, 1.0 - _EPS)
    classes = (totals["classes"] / count).clamp_min(_EPS)
    return MultitaskOutput(
        tissue_logits=tissue.log(),
        nuclei_center_logits=center.log() - (1.0 - center).log(),
        nuclei_class_logits=classes.log(),
        nuclei_offsets=totals["offsets"] / count,
        nuclei_sizes=totals["sizes"] / count,
        auxiliary={},
    )
