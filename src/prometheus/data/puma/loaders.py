"""Reproducible task-specific PUMA data loaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from .datasets import PUMADataset
from .splits import load_or_create_split


def collate_segmentation(batch):
    images = torch.stack([item[0] for item in batch])
    targets = {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1]}
    return images, targets


def collate_detection(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def create_puma_dataloaders(
    root,
    image_size: int = 1024,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
    train_transforms=None,
    val_transforms=None,
    test_split: float | None = None,
    test_transforms=None,
    stratified_split: bool = True,
    split_manifest_path=None,
    pin_memory: bool | None = None,
):
    """Create legacy dual-task loaders with a reproducible split."""

    if not stratified_split:
        split_manifest_path = None
    validation_fraction = test_split if test_split is not None else val_split
    validation_transforms = test_transforms if test_transforms is not None else val_transforms
    train_dataset = PUMADataset(root, image_size, transforms=train_transforms)
    validation_dataset = PUMADataset(
        root,
        image_size,
        augment=False,
        transforms=validation_transforms,
    )
    train_ids, validation_ids = load_or_create_split(
        train_dataset.samples,
        validation_fraction,
        seed,
        split_manifest_path,
    )
    index_by_id = {sample.sample_id: index for index, sample in enumerate(train_dataset.samples)}
    train_indices = [index_by_id[sample_id] for sample_id in train_ids]
    validation_indices = [index_by_id[sample_id] for sample_id in validation_ids]
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_segmentation,
        "pin_memory": bool(pin_memory),
    }
    train_loader = DataLoader(Subset(train_dataset, train_indices), shuffle=True, **common)
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader
