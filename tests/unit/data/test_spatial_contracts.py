import numpy as np

from prometheus.data.spatial import boxes_to_model, boxes_to_source, letterbox_image, points_to_model, points_to_source
from prometheus.data.transforms import NormalizeMultitask, RandomRotate90Multitask, TransformSample


def test_letterbox_coordinate_round_trip() -> None:
    _, meta = letterbox_image(np.zeros((40, 80, 3), dtype=np.float32), (64, 64), "sample")
    points = np.array([[0.0, 0.0], [79.0, 39.0], [20.5, 10.25]], dtype=np.float32)
    np.testing.assert_allclose(points_to_source(points_to_model(points, meta), meta), points, atol=1e-5)
    assert meta.resized_size == (32, 64)
    assert meta.pad_xy == (0, 16)


def test_box_coordinate_round_trip() -> None:
    _, meta = letterbox_image(np.zeros((30, 60, 3), dtype=np.float32), (64, 64), "sample")
    boxes = np.array([[2.0, 3.0, 20.0, 25.0]], dtype=np.float32)
    np.testing.assert_allclose(boxes_to_source(boxes_to_model(boxes, meta), meta), boxes, atol=1e-5)


def test_rotate90_updates_all_geometry(monkeypatch) -> None:
    monkeypatch.setattr("prometheus.data.transforms.multitask.random.choice", lambda _: 1)
    sample = TransformSample(
        image=np.zeros((3, 8, 8), dtype=np.float32),
        tissue_mask=np.zeros((8, 8), dtype=np.uint8),
        centroids=np.array([[2.0, 3.0]], dtype=np.float32),
        boxes=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    )
    rotated = RandomRotate90Multitask()(sample)
    np.testing.assert_allclose(rotated.centroids, [[3.0, 5.0]])
    np.testing.assert_allclose(rotated.boxes, [[2.0, 4.0, 4.0, 6.0]])


def test_normalization_excludes_and_preserves_letterbox_padding() -> None:
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[:, 2:6] = np.linspace(0, 1, 32, dtype=np.float32).reshape(1, 4, 8)
    valid_mask = np.zeros((8, 8), dtype=bool)
    valid_mask[2:6] = True
    sample = TransformSample(
        image,
        np.zeros((8, 8), dtype=np.uint8),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        valid_mask,
    )
    normalized = NormalizeMultitask()(sample)
    assert np.all(normalized.image[:, ~valid_mask] == 0)
    np.testing.assert_allclose(normalized.image[:, valid_mask].mean(axis=1), 0, atol=1e-5)
