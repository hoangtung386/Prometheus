"""Reproducible task-specific PUMA data loaders."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..collate import collate_multitask
from ..transforms.multitask import multitask_train_transform, multitask_validation_transform
from .multitask_dataset import PumaMultitaskDataset
from .splits import load_or_create_kfold, load_or_create_split


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_multitask_dataloaders(
    root,
    image_size: tuple[int, int] | int = (1024, 1024),
    batch_size: int = 4,
    num_workers: int = 4,
    validation_fraction: float = 0.1,
    seed: int = 42,
    split_manifest_path=None,
    pin_memory: bool = True,
    strict_labels: bool = True,
):
    """Create instance-aware loaders for the refactored multitask model."""
    train_dataset = PumaMultitaskDataset(
        root,
        image_size,
        transforms=multitask_train_transform(),
        strict_labels=strict_labels,
    )
    validation_dataset = PumaMultitaskDataset(
        root,
        image_size,
        transforms=multitask_validation_transform(),
        strict_labels=strict_labels,
    )
    train_ids, validation_ids = load_or_create_split(
        train_dataset.samples,
        validation_fraction,
        seed,
        split_manifest_path,
    )
    return _build_multitask_loaders(
        train_dataset, validation_dataset, train_ids, validation_ids,
        batch_size, num_workers, seed, pin_memory,
    )


def create_multitask_kfold_dataloaders(
    root,
    image_size: tuple[int, int] | int = (1024, 1024),
    batch_size: int = 4,
    num_workers: int = 4,
    num_folds: int = 5,
    fold_index: int = 0,
    seed: int = 42,
    kfold_manifest_path=None,
    pin_memory: bool = True,
    strict_labels: bool = True,
):
    """Instance-aware loaders for one fold of a k-fold cross-validation over all data."""
    if not 0 <= fold_index < num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds})")
    train_dataset = PumaMultitaskDataset(
        root, image_size, transforms=multitask_train_transform(), strict_labels=strict_labels
    )
    validation_dataset = PumaMultitaskDataset(
        root, image_size, transforms=multitask_validation_transform(), strict_labels=strict_labels
    )
    folds = load_or_create_kfold(train_dataset.samples, num_folds, seed, kfold_manifest_path)
    train_ids, validation_ids = folds[fold_index]
    return _build_multitask_loaders(
        train_dataset, validation_dataset, train_ids, validation_ids,
        batch_size, num_workers, seed, pin_memory,
    )


def _build_multitask_loaders(
    train_dataset,
    validation_dataset,
    train_ids,
    validation_ids,
    batch_size,
    num_workers,
    seed,
    pin_memory,
):
    index_by_id = {sample.sample_id: index for index, sample in enumerate(train_dataset.samples)}
    train_indices = [index_by_id[sample_id] for sample_id in train_ids]
    validation_indices = [index_by_id[sample_id] for sample_id in validation_ids]
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_multitask,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
    }
    train_generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(Subset(train_dataset, train_indices), shuffle=True, generator=train_generator, **common),
        DataLoader(Subset(validation_dataset, validation_indices), shuffle=False, **common),
    )
