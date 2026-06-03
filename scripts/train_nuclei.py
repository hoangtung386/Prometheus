#!/usr/bin/env python3
"""Training script for DualUNet (nuclei stream)."""
from __future__ import annotations

import torch
import torch.optim as optim

from prometheus import CombinedLoss, DualUNet
from prometheus.config import ModelConfig, TrainingConfig


def main() -> None:
    model_cfg = ModelConfig(in_chans=3, num_classes=1)
    train_cfg = TrainingConfig(batch_size=2, epochs=5)

    model = DualUNet(config=model_cfg)
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
    dummy_target_tissue = torch.randn(train_cfg.batch_size, 1, 256, 256)
    dummy_target_nuclei = torch.randn(train_cfg.batch_size, 1, 256, 256)

    for step in range(train_cfg.epochs):
        optimizer.zero_grad()
        tissue_mask, nuclei_mask, moe_loss = model(dummy_input)
        loss_tissue = criterion(tissue_mask, dummy_target_tissue)
        loss_nuclei = criterion(nuclei_mask, dummy_target_nuclei)
        loss = loss_tissue + loss_nuclei + 0.01 * moe_loss
        loss.backward()
        optimizer.step()
        print(
            f"Step {step + 1}: loss = {loss.item():.6f} "
            f"(tissue={loss_tissue.item():.6f}, "
            f"nuclei={loss_nuclei.item():.6f}, "
            f"moe={moe_loss.item():.6f})"
        )


if __name__ == "__main__":
    main()
