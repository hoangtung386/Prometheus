#!/usr/bin/env python3
"""Training script for PUMA dataset with multiclass segmentation.

Usage:
    python scripts/train_puma.py --data-root /path/to/puma --model-type DualUNet

Supports UNetTissue (tissue only) and DualUNet (tissue + nuclei).
Uses MulticlassCombinedLoss (CrossEntropy + MultiClassDice) and logs per-class Dice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from prometheus import DualUNet, Trainer, UNetTissue
from prometheus.config import ModelConfig, TrainingConfig
from prometheus.data import create_puma_dataloaders, test_transform, train_transform
from prometheus.data.puma_dataset import NUCLEI_CLASSES, TISSUE_CLASSES


def _print_class_stats(
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    modality: str,
) -> None:
    pixel_counts: Counter[int] = Counter()
    total_pixels = 0
    for _, targets in loader:
        mask = targets[modality]
        for cls_idx in range(len(class_names)):
            pixel_counts[cls_idx] += (mask == cls_idx).sum().item()
        total_pixels += mask.numel()
    print(f"  [{modality}] Pixel distribution ({total_pixels:,} total):")
    for cls_idx in range(len(class_names)):
        count = pixel_counts[cls_idx]
        ratio = count / max(total_pixels, 1) * 100
        print(f"    {class_names[cls_idx]}: {ratio:.2f}% ({count:,} px)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PUMA segmentation model")
    parser.add_argument("--data-root", type=str, required=True, help="Path to PUMA dataset root")
    parser.add_argument("--model-type", type=str, default="DualUNet", choices=["UNetTissue", "DualUNet"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="logs/puma")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/puma")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-split", type=float, default=0.1)
    args = parser.parse_args()

    num_tissue_classes = len(TISSUE_CLASSES)
    num_nuclei_classes = len(NUCLEI_CLASSES)

    train_cfg = TrainingConfig(
        model_type=args.model_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        num_tissue_classes=num_tissue_classes,
        num_nuclei_classes=num_nuclei_classes,
        data_root=args.data_root,
        num_workers=args.num_workers,
        test_split=args.test_split,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
    )

    if args.model_type == "UNetTissue":
        model_cfg = ModelConfig(in_chans=3, num_tissue_classes=num_tissue_classes)
        model = UNetTissue(config=model_cfg)
    else:
        model_cfg = ModelConfig(
            in_chans=3,
            num_tissue_classes=num_tissue_classes,
            num_nuclei_classes=num_nuclei_classes,
        )
        model = DualUNet(config=model_cfg)

    print(f"Model: {args.model_type}")
    print(f"  Tissue classes: {num_tissue_classes}, Nuclei classes: {num_nuclei_classes}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} ({trainable:,} trainable)")

    train_loader, test_loader = create_puma_dataloaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_split=args.test_split,
        train_transforms=train_transform(),
        test_transforms=test_transform(),
    )

    print(f"\nDataset:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples:  {len(test_loader.dataset)}")

    print("\nTarget class distribution (train set):")
    _print_class_stats(train_loader, TISSUE_CLASSES, "tissue")
    if args.model_type != "UNetTissue":
        _print_class_stats(train_loader, NUCLEI_CLASSES, "nuclei")

    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=train_cfg,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
