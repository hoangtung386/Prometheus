import torch

from prometheus.data.targets import encode_centerpoint_targets
from prometheus.domain import NucleiTarget


def test_touching_instances_remain_distinct_targets() -> None:
    target = NucleiTarget(
        centroids=torch.tensor([[10.0, 10.0], [14.0, 10.0]]),
        labels=torch.tensor([2, 2]),
        boxes=torch.tensor([[8.0, 8.0, 12.0, 12.0], [12.0, 8.0, 16.0, 12.0]]),
    )
    encoded = encode_centerpoint_targets([target], (16, 16), stride=1, num_classes=10)
    assert encoded.indices[0].tolist() == [170, 174]
    assert encoded.heatmap[0, 2, 10, 10] == 1
    assert encoded.heatmap[0, 2, 10, 14] == 1
