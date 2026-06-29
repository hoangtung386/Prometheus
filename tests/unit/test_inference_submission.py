import json

import numpy as np
import pytest
import torch

from prometheus.inference import PredictionPipeline
from prometheus.io import write_nuclei_json, write_tissue_tiff
from prometheus.submission import validate_submission_outputs


class _FakeDualModel(torch.nn.Module):
    def forward(self, images):
        batch_size, _, height, width = images.shape
        tissue = torch.zeros(batch_size, 6, height, width)
        nuclei = torch.zeros(batch_size, 11, height, width)
        tissue[:, 1] = 5
        nuclei[:, 1, 1:3, 1:3] = 5
        return tissue, nuclei, torch.tensor(0.0)


def test_prediction_pipeline_converts_semantic_components_to_detections() -> None:
    result = PredictionPipeline(_FakeDualModel()).predict(torch.zeros(1, 3, 4, 4))
    assert result.tissue_mask.shape == (1, 4, 4)
    assert len(result.nuclei[0]) == 1
    assert result.nuclei[0][0].centroid == (1.5, 1.5)


def test_submission_validator_accepts_generated_outputs(tmp_path) -> None:
    tissue_path = tmp_path / "tissue.tif"
    nuclei_path = tmp_path / "nuclei.json"
    write_tissue_tiff(np.zeros((4, 4), dtype=np.uint8), tissue_path)
    write_nuclei_json([], nuclei_path)
    validate_submission_outputs(tissue_path, nuclei_path)


def test_submission_validator_rejects_wrong_nuclei_schema(tmp_path) -> None:
    tissue_path = tmp_path / "tissue.tif"
    nuclei_path = tmp_path / "nuclei.json"
    write_tissue_tiff(np.zeros((4, 4), dtype=np.uint8), tissue_path)
    nuclei_path.write_text(json.dumps({"features": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="polygons"):
        validate_submission_outputs(tissue_path, nuclei_path)
