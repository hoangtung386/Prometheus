"""Dihedral test-time augmentation.

The offset field is a signed direction, so a view must rotate its *components* as well as
resample its grid. Getting that wrong shifts every decoded centroid by up to a stride,
which is invisible in the loss and shows up only as degraded nuclei F1.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from prometheus.inference.tta import (
    DIHEDRAL_VIEWS,
    FLIP_VIEWS,
    _forward_offsets,
    _invert_offsets,
    tta_forward,
)
from prometheus.models import MultitaskOutput

NUM_TISSUE_CLASSES = 6
NUM_NUCLEUS_CLASSES = 10


@pytest.mark.parametrize(("rotations", "flip"), DIHEDRAL_VIEWS)
def test_offset_transform_round_trips(rotations: int, flip: bool) -> None:
    torch.manual_seed(0)
    offsets = torch.randn(2, 2, 8, 8)
    restored = _invert_offsets(_forward_offsets(offsets, rotations, flip), rotations, flip)

    assert torch.allclose(restored, offsets, atol=1e-6)


def test_dihedral_views_are_the_eight_distinct_group_elements() -> None:
    assert len(set(DIHEDRAL_VIEWS)) == 8
    assert set(FLIP_VIEWS) <= set(DIHEDRAL_VIEWS)


class _EquivariantModel(nn.Module):
    """A per-pixel model, hence exactly equivariant under any dihedral view.

    Averaging an equivariant model over the group must be a no-op, which makes this the
    tightest available check that every inverse transform is correct.
    """

    def forward(self, images: torch.Tensor) -> MultitaskOutput:
        batch, _, height, width = images.shape
        channel = images[:, :1]
        return MultitaskOutput(
            tissue_logits=channel.repeat(1, NUM_TISSUE_CLASSES, 1, 1)
            * torch.arange(1, NUM_TISSUE_CLASSES + 1).view(1, -1, 1, 1),
            nuclei_center_logits=channel,
            nuclei_class_logits=channel.repeat(1, NUM_NUCLEUS_CLASSES, 1, 1),
            nuclei_offsets=torch.zeros(batch, 2, height, width),
            nuclei_sizes=channel.abs().repeat(1, 2, 1, 1),
        )


def test_averaging_an_equivariant_model_is_a_no_op() -> None:
    torch.manual_seed(0)
    model = _EquivariantModel()
    images = torch.randn(2, 3, 8, 8)
    plain = model(images)
    averaged = tta_forward(model, images, DIHEDRAL_VIEWS)

    assert torch.allclose(averaged.tissue_logits.softmax(1), plain.tissue_logits.softmax(1), atol=1e-5)
    assert torch.allclose(averaged.nuclei_center_logits.sigmoid(), plain.nuclei_center_logits.sigmoid(), atol=1e-5)
    assert torch.allclose(averaged.nuclei_sizes, plain.nuclei_sizes, atol=1e-5)


def test_output_shapes_are_preserved() -> None:
    output = tta_forward(_EquivariantModel(), torch.randn(1, 3, 8, 8), FLIP_VIEWS)

    assert output.tissue_logits.shape == (1, NUM_TISSUE_CLASSES, 8, 8)
    assert output.nuclei_class_logits.shape == (1, NUM_NUCLEUS_CLASSES, 8, 8)
    assert output.nuclei_offsets.shape == (1, 2, 8, 8)


def test_empty_view_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="view"):
        tta_forward(_EquivariantModel(), torch.randn(1, 3, 8, 8), ())
