#!/usr/bin/env python3
"""Training script for UNetTissue model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from prometheus import UNetTissue
from prometheus.config import ModelConfig


def main() -> None:
    cfg = ModelConfig(in_chans=3, num_classes=1)
    model = UNetTissue(config=cfg)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    dummy_input = torch.randn(4, 3, 256, 256)
    dummy_target = torch.randn(4, 1, 256, 256)

    for step in range(10):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}: loss = {loss.item():.6f}")


if __name__ == "__main__":
    main()
