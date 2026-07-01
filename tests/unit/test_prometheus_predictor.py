import numpy as np
import torch

from prometheus.domain import ImageMeta, NucleusClass
from prometheus.inference import PrometheusPredictor
from prometheus.models import MultitaskOutput


class _FakePrometheus(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> MultitaskOutput:
        tissue = torch.zeros(images.shape[0], 6, *images.shape[-2:], device=images.device)
        tissue[:, 2] = 5
        centers = torch.full((images.shape[0], 10, 8, 8), -10.0, device=images.device)
        centers[:, 3, 2, 4] = 10.0
        classes = torch.full_like(centers, -5.0)
        classes[:, 3, 2, 4] = 5.0
        offsets = torch.zeros(images.shape[0], 2, 8, 8, device=images.device)
        sizes = torch.ones(images.shape[0], 2, 8, 8, device=images.device) * 2
        return MultitaskOutput(tissue, centers, classes, offsets, sizes)


def test_predictor_restores_source_space() -> None:
    meta = ImageMeta("x", (16, 32), (32, 32), (16, 32), (1.0, 1.0), (0, 8))
    result = PrometheusPredictor(_FakePrometheus(), nuclei_stride=4).predict(torch.zeros(1, 3, 32, 32), [meta])
    assert result.tissue_masks[0].shape == (16, 32)
    assert np.all(result.tissue_masks[0] == 2)
    assert len(result.nuclei[0]) == 1
    assert result.nuclei[0][0].label is NucleusClass.HISTIOCYTE
    assert result.nuclei[0][0].centroid == (16.0, 0.0)
