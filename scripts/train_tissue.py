#!/usr/bin/env python3
"""Training script for UNetTissue model."""
from __future__ import annotations

import torch
import torch.optim as optim

from prometheus import CombinedLoss, UNetTissue
from prometheus.config import ModelConfig, TrainingConfig


def main() -> None:
    model_cfg = ModelConfig(in_chans=3, num_classes=1)
    train_cfg = TrainingConfig(batch_size=4, epochs=10)

    model = UNetTissue(config=model_cfg)
    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
        eps=train_cfg.eps,
    )
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)

    dummy_input = torch.randn(train_cfg.batch_size, 3, 256, 256)
    dummy_target = torch.randn(train_cfg.batch_size, 1, 256, 256)

    for step in range(train_cfg.epochs):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}: loss = {loss.item():.6f}")


if __name__ == "__main__":
    main()
