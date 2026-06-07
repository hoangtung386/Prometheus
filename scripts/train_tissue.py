#!/usr/bin/env python3
"""Training script for UNetTissue model."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from prometheus import CombinedLoss, Trainer, UNetTissue
from prometheus.config import ModelConfig, TrainingConfig


def main() -> None:
    model_cfg = ModelConfig(in_chans=3, num_classes=1)
    train_cfg = TrainingConfig(
        model_type="UNetTissue",
        batch_size=4,
        epochs=10,
        log_dir="logs/tissue",
        ckpt_dir="checkpoints/tissue",
    )

    model = UNetTissue(config=model_cfg)

    dummy = [
        (torch.randn(3, 256, 256),
         {"tissue": torch.randint(0, 2, (1, 256, 256)).float(),
          "nuclei": torch.randint(0, 2, (1, 256, 256)).float()})
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

