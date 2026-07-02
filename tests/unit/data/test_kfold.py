from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.data.puma.splits import create_kfold, load_or_create_kfold
from prometheus.domain import PumaSample


def _samples(count: int) -> list[PumaSample]:
    samples = []
    for index in range(count):
        group = "primary" if index % 2 == 0 else "metastatic"
        sample_id = f"training_set_{group}_roi_{index:03d}"
        samples.append(PumaSample(sample_id, Path("i"), Path("t"), Path("n")))
    return samples


def test_kfold_validates_every_sample_exactly_once() -> None:
    samples = _samples(205)
    folds = create_kfold(samples, num_folds=5, seed=42)
    assert len(folds) == 5
    validation_union = sorted(sid for _, val_ids in folds for sid in val_ids)
    assert validation_union == sorted(s.sample_id for s in samples)  # full coverage, no overlap
    sizes = [len(val_ids) for _, val_ids in folds]
    assert max(sizes) - min(sizes) <= 2  # balanced folds
    for train_ids, val_ids in folds:
        assert set(train_ids).isdisjoint(val_ids)
        assert len(train_ids) + len(val_ids) == 205


def test_kfold_is_deterministic() -> None:
    samples = _samples(40)
    assert create_kfold(samples, 5, 7) == create_kfold(samples, 5, 7)


def test_kfold_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        create_kfold(_samples(10), num_folds=1, seed=42)
    with pytest.raises(ValueError):
        create_kfold(_samples(3), num_folds=5, seed=42)


def test_load_or_create_kfold_caches_and_reloads(tmp_path: Path) -> None:
    samples = _samples(30)
    manifest = tmp_path / "kfold.json"
    created = load_or_create_kfold(samples, 5, 42, manifest)
    assert manifest.is_file()
    assert load_or_create_kfold(samples, 5, 42, manifest) == created  # served from cache
