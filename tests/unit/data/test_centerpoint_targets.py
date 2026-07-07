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


def test_class_agnostic_heatmap_preserves_labels_for_classifier() -> None:
    target = NucleiTarget(
        centroids=torch.tensor([[4.0, 5.0], [9.0, 7.0]]),
        labels=torch.tensor([2, 8]),
        boxes=torch.tensor([[2.0, 3.0, 6.0, 7.0], [7.0, 5.0, 11.0, 9.0]]),
    )
    encoded = encode_centerpoint_targets(
        [target], (16, 16), stride=1, num_classes=10, class_agnostic=True
    )
    assert encoded.heatmap.shape == (1, 1, 16, 16)
    assert encoded.heatmap[0, 0, 5, 4] == 1
    assert encoded.heatmap[0, 0, 7, 9] == 1
    assert encoded.labels[0].tolist() == [2, 8]
