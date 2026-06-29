"""Create and persist deterministic PUMA train/validation splits."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ...domain import PumaSample

SPLIT_SCHEMA_VERSION = 1


def _sample_group(sample_id: str) -> str:
    lowered = sample_id.lower()
    if "primary" in lowered:
        return "primary"
    if "metastatic" in lowered:
        return "metastatic"
    return "other"


def create_split(
    samples: list[PumaSample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if len(samples) < 2:
        raise ValueError("At least two samples are required to create a split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(_sample_group(sample.sample_id), []).append(sample.sample_id)
    random_generator = random.Random(seed)
    validation_ids = []
    for group_ids in groups.values():
        ordered_ids = sorted(group_ids)
        random_generator.shuffle(ordered_ids)
        group_count = round(len(ordered_ids) * validation_fraction)
        if len(ordered_ids) > 1:
            group_count = max(1, min(group_count, len(ordered_ids) - 1))
        validation_ids.extend(ordered_ids[:group_count])
    if not validation_ids:
        validation_ids.append(sorted(sample.sample_id for sample in samples)[0])
    validation_set = set(validation_ids)
    train_ids = sorted(sample.sample_id for sample in samples if sample.sample_id not in validation_set)
    return train_ids, sorted(validation_set)


def load_or_create_split(
    samples: list[PumaSample],
    validation_fraction: float,
    seed: int,
    manifest_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    path = Path(manifest_path) if manifest_path else None
    current_ids = {sample.sample_id for sample in samples}
    if path is not None and path.is_file():
        with path.open(encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported split manifest schema: {path}")
        train_ids = manifest.get("train", [])
        validation_ids = manifest.get("validation", [])
        manifest_ids = set(train_ids) | set(validation_ids)
        if manifest_ids != current_ids or set(train_ids) & set(validation_ids):
            raise ValueError(f"Split manifest does not match the discovered dataset: {path}")
        return sorted(train_ids), sorted(validation_ids)

    train_ids, validation_ids = create_split(samples, validation_fraction, seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "schema_version": SPLIT_SCHEMA_VERSION,
                    "seed": seed,
                    "validation_fraction": validation_fraction,
                    "train": train_ids,
                    "validation": validation_ids,
                },
                file_obj,
                indent=2,
            )
    return train_ids, validation_ids
