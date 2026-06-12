#!/usr/bin/env python3
"""Smoke test training script for UNetTissue model (multiclass)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from prometheus import Trainer, UNetTissue
from prometheus.config import ModelConfig, TrainingConfig


def main() -> None:
    num_classes = 6
    model_cfg = ModelConfig(in_chans=3, num_classes=num_classes)
    train_cfg = TrainingConfig(
        model_type="UNetTissue",
        num_tissue_classes=num_classes,
        batch_size=4,
        epochs=10,
        log_dir="logs/tissue",
        ckpt_dir="checkpoints/tissue",
    )

    model = UNetTissue(config=model_cfg)

    dummy = [
        (torch.randn(3, 256, 256),
         {"tissue": torch.randint(0, num_classes, (256, 256)).long(),
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

