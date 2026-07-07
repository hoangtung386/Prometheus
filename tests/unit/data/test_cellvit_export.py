import numpy as np

from prometheus.data.puma.cellvit_export import CELLVIT_TYPE_ID, rasterize_cellvit_instances
from prometheus.domain import NucleusClass, NucleusInstance


def _instance(label: NucleusClass, polygon: list[list[float]]) -> NucleusInstance:
    points = np.asarray(polygon, dtype=np.float32)
    return NucleusInstance("x", label, points, (2.0, 2.0), (1.0, 1.0, 3.0, 3.0))


def test_cellvit_export_uses_official_track2_type_order() -> None:
    assert CELLVIT_TYPE_ID[NucleusClass.ENDOTHELIUM] == 1
    assert CELLVIT_TYPE_ID[NucleusClass.TUMOR] == 4
    assert CELLVIT_TYPE_ID[NucleusClass.LYMPHOCYTE] == 10


def test_cellvit_export_preserves_instances_and_types() -> None:
    instances = [
        _instance(NucleusClass.TUMOR, [[1, 1], [3, 1], [3, 3], [1, 3]]),
        _instance(NucleusClass.LYMPHOCYTE, [[5, 5], [7, 5], [7, 7], [5, 7]]),
    ]
    instance_map, type_map = rasterize_cellvit_instances(instances, (10, 10))
    assert set(np.unique(instance_map)) == {0, 1, 2}
    assert type_map[2, 2] == 4
    assert type_map[6, 6] == 10
