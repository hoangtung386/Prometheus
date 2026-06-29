from pathlib import Path

from prometheus.data.puma.splits import load_or_create_split
from prometheus.domain import PumaSample


def _sample(sample_id: str) -> PumaSample:
    path = Path(f"/{sample_id}")
    return PumaSample(sample_id, path, path, path)


def test_split_manifest_is_persisted_and_reused(tmp_path) -> None:
    samples = [
        _sample("primary_001"),
        _sample("primary_002"),
        _sample("metastatic_001"),
        _sample("metastatic_002"),
    ]
    manifest = tmp_path / "split.json"
    first = load_or_create_split(samples, 0.25, 42, manifest)
    second = load_or_create_split(list(reversed(samples)), 0.25, 999, manifest)
    assert first == second
    assert not set(first[0]) & set(first[1])
