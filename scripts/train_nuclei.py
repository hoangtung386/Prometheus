#!/usr/bin/env python3
"""Smoke test training script for DualUNet (multiclass tissue + nuclei)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from prometheus import DualUNet, Trainer
from prometheus.config import ModelConfig, TrainingConfig


def main() -> None:
    model_cfg = ModelConfig(in_chans=3, num_tissue_classes=6, num_nuclei_classes=11)
    train_cfg = TrainingConfig(
        model_type="DualUNet",
        num_tissue_classes=6,
        num_nuclei_classes=11,
        batch_size=2,
        epochs=5,
        log_dir="logs/nuclei",
        ckpt_dir="checkpoints/nuclei",
    )

    model = DualUNet(config=model_cfg)

    dummy = [
        (torch.randn(3, 256, 256),
         {"tissue": torch.randint(0, 6, (256, 256)).long(),
          "nuclei": torch.randint(0, 11, (256, 256)).long()})
        for _ in range(64)
    ]
    train_loader = val_loader = DataLoader(dummy, batch_size=train_cfg.batch_size, shuffle=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
    )
    trainer.fit()


if __name__ == "__main__":
    main()

