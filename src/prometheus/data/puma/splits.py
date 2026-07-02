"""Create and persist deterministic PUMA train/validation splits."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ...domain import PumaSample

SPLIT_SCHEMA_VERSION = 1
KFOLD_SCHEMA_VERSION = 1


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


def create_kfold(
    samples: list[PumaSample],
    num_folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    """Partition every sample into ``num_folds`` folds; return per-fold (train, val) ids.

    Assignment is round-robin within each primary/metastatic group so folds stay class
    balanced, and deterministic given ``seed``. Each sample is a validation item in
    exactly one fold, so the union of validation sets covers the whole dataset.
    """
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    if len(samples) < num_folds:
        raise ValueError("Need at least num_folds samples to build the folds")
    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(_sample_group(sample.sample_id), []).append(sample.sample_id)
    random_generator = random.Random(seed)
    fold_validation: list[list[str]] = [[] for _ in range(num_folds)]
    for group_ids in groups.values():
        ordered_ids = sorted(group_ids)
        random_generator.shuffle(ordered_ids)
        for position, sample_id in enumerate(ordered_ids):
            fold_validation[position % num_folds].append(sample_id)
    all_ids = sorted(sample.sample_id for sample in samples)
    folds: list[tuple[list[str], list[str]]] = []
    for fold_index in range(num_folds):
        validation_set = set(fold_validation[fold_index])
        train_ids = sorted(sample_id for sample_id in all_ids if sample_id not in validation_set)
        validation_ids = sorted(validation_set)
        if not validation_ids or not train_ids:
            raise ValueError(f"Fold {fold_index} is degenerate; reduce num_folds")
        folds.append((train_ids, validation_ids))
    return folds


def load_or_create_kfold(
    samples: list[PumaSample],
    num_folds: int,
    seed: int,
    manifest_path: str | Path | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Return the cached k-fold partition, or create and persist it once."""
    path = Path(manifest_path) if manifest_path else None
    current_ids = {sample.sample_id for sample in samples}
    if path is not None and path.is_file():
        with path.open(encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        if manifest.get("schema_version") != KFOLD_SCHEMA_VERSION or manifest.get("num_folds") != num_folds:
            raise ValueError(f"Incompatible k-fold manifest (schema/num_folds): {path}")
        folds = [(sorted(fold["train"]), sorted(fold["validation"])) for fold in manifest["folds"]]
        manifest_ids = {sample_id for train_ids, val_ids in folds for sample_id in (*train_ids, *val_ids)}
        if manifest_ids != current_ids:
            raise ValueError(f"K-fold manifest does not match the discovered dataset: {path}")
        return folds

    folds = create_kfold(samples, num_folds, seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "schema_version": KFOLD_SCHEMA_VERSION,
                    "seed": seed,
                    "num_folds": num_folds,
                    "folds": [{"train": train_ids, "validation": val_ids} for train_ids, val_ids in folds],
                },
                file_obj,
                indent=2,
            )
    return folds
