import json

import numpy as np
import pytest

from prometheus.io import write_nuclei_json, write_tissue_tiff
from prometheus.submission import validate_submission_outputs


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
