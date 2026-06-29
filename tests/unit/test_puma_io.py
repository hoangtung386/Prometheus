import json

import numpy as np
import tifffile

from prometheus.domain import Detection, NucleusClass, Track
from prometheus.io import write_nuclei_json, write_tissue_tiff


def test_nuclei_writer_emits_official_multiple_polygons_schema(tmp_path) -> None:
    path = tmp_path / "nuclei.json"
    write_nuclei_json(
        [Detection((10.0, 20.0), NucleusClass.TUMOR, confidence=0.75)],
        path,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    polygon = data["polygons"][0]
    assert polygon["name"] == "nuclei_tumor"
    assert len(polygon["path_points"]) == 4
    assert polygon["score"] == 0.75


def test_track_one_merges_plasma_cells_into_lymphocytes(tmp_path) -> None:
    path = tmp_path / "nuclei.json"
    write_nuclei_json(
        [Detection((1.0, 2.0), NucleusClass.PLASMA_CELL)],
        path,
        track=Track.TRACK_1,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["polygons"][0]["name"] == "nuclei_lymphocyte"


def test_tissue_writer_remaps_classes_and_writes_required_tags(tmp_path) -> None:
    path = tmp_path / "tissue.tif"
    # Internal order: background, tumor, stroma, epidermis, necrosis, vessel.
    write_tissue_tiff(np.array([[0, 1, 2, 3, 4, 5]], dtype=np.uint8), path)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.tags["XResolution"].value
        assert page.tags["YResolution"].value
        assert page.tags["SMinSampleValue"].value == 1
        assert page.tags["SMaxSampleValue"].value == 5
        assert page.asarray().tolist() == [[0, 3, 1, 4, 5, 2]]
