import numpy as np
import pytest
import torch

from prometheus.domain import ImageMeta, NucleusClass
from prometheus.inference import DIHEDRAL_VIEWS, PrometheusPredictor
from prometheus.models import MultitaskOutput

_STRIDE = 4
_MAP_SIZE = 8
_PEAK = (2, 4)  # (row, column) in the stride-4 nuclei map


class _FakePrometheus(torch.nn.Module):
    """One tissue class everywhere and a single nucleus peak, independent of the input."""

    def forward(self, images: torch.Tensor) -> MultitaskOutput:
        batch, device = images.shape[0], images.device
        tissue = torch.zeros(batch, 6, *images.shape[-2:], device=device)
        tissue[:, 2] = 5.0
        centers = torch.full((batch, 1, _MAP_SIZE, _MAP_SIZE), -10.0, device=device)
        centers[:, 0, _PEAK[0], _PEAK[1]] = 10.0
        classes = torch.full((batch, 10, _MAP_SIZE, _MAP_SIZE), -5.0, device=device)
        classes[:, 3, _PEAK[0], _PEAK[1]] = 5.0
        return MultitaskOutput(
            tissue_logits=tissue,
            nuclei_center_logits=centers,
            nuclei_class_logits=classes,
            nuclei_offsets=torch.zeros(batch, 2, _MAP_SIZE, _MAP_SIZE, device=device),
            nuclei_sizes=torch.full((batch, 2, _MAP_SIZE, _MAP_SIZE), 2.0, device=device),
        )


def _letterboxed_meta() -> ImageMeta:
    """A 16x32 source image letterboxed into a 32x32 model input with 8px top/bottom pads."""
    return ImageMeta("x", (16, 32), (32, 32), (16, 32), (1.0, 1.0), (0, 8))


def test_predictor_restores_source_space() -> None:
    predictor = PrometheusPredictor(_FakePrometheus(), nuclei_stride=_STRIDE, tta_views=())
    result = predictor.predict(torch.zeros(1, 3, 32, 32), [_letterboxed_meta()])

    assert result.tissue_masks[0].shape == (16, 32), "the mask must be cropped back to the source size"
    assert np.all(result.tissue_masks[0] == 2)
    assert len(result.nuclei[0]) == 1
    assert result.nuclei[0][0].label is NucleusClass.HISTIOCYTE
    # Model space (4*4, 4*2) = (16, 8); undoing an 8px vertical pad puts it at y = 0.
    assert result.nuclei[0][0].centroid == (16.0, 0.0)


def test_predictor_defaults_to_full_dihedral_tta() -> None:
    # The submission path is where averaging is free accuracy, so it must be on by default.
    assert PrometheusPredictor(_FakePrometheus()).tta_views == tuple(DIHEDRAL_VIEWS)


def test_predictor_rejects_a_class_specific_center_map() -> None:
    class _LegacyHead(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> MultitaskOutput:
            output = _FakePrometheus()(images)
            return MultitaskOutput(
                tissue_logits=output.tissue_logits,
                nuclei_center_logits=output.nuclei_center_logits.repeat(1, 10, 1, 1),
                nuclei_class_logits=output.nuclei_class_logits,
                nuclei_offsets=output.nuclei_offsets,
                nuclei_sizes=output.nuclei_sizes,
            )

    predictor = PrometheusPredictor(_LegacyHead(), nuclei_stride=_STRIDE, tta_views=())
    with pytest.raises(ValueError, match="class-agnostic"):
        predictor.predict(torch.zeros(1, 3, 32, 32), [_letterboxed_meta()])
