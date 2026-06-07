from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..losses import CombinedLoss


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.argmax(dim=1)
    pred = nn.functional.one_hot(pred, num_classes=target.shape[1]).permute(0, 3, 1, 2)
    intersection = (pred * target).sum(dim=(2, 3))
    cardinality = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * intersection + eps) / (cardinality + eps)).mean(dim=1)


def warmup_cosine_lr(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        t_mask = targets["tissue"].to(device, non_blocking=True)
        n_mask = targets["nuclei"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=config.amp):
            if config.model_type == "UNetTissue":
                logits = model(images)
                loss = criterion(logits, t_mask)
            else:
                pred_t, pred_n, moe_loss = model(images)
                loss = (
                    criterion(pred_t, t_mask)
                    + criterion(pred_n, n_mask)
                    + config.moe_loss_weight * moe_loss
                )

        scaler.scale(loss).backward()
        if config.gradient_clip_norm and config.gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % config.log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"E{epoch:03d} B{batch_idx:04d}/{n}  loss={loss.item():.4f}  lr={lr_now:.2e}")

    return total_loss / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    dice_tissue, dice_nuclei = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        t_mask = targets["tissue"].to(device, non_blocking=True)
        n_mask = targets["nuclei"].to(device, non_blocking=True)

        if config.model_type == "UNetTissue":
            logits = model(images)
            loss = criterion(logits, t_mask)
            dice_tissue.append(dice_score(logits, t_mask))
        else:
            pred_t, pred_n, _ = model(images)
            loss = criterion(pred_t, t_mask) + criterion(pred_n, n_mask)
            dice_tissue.append(dice_score(pred_t, t_mask))
            dice_nuclei.append(dice_score(pred_n, n_mask))

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    avg_dice_t = torch.cat(dice_tissue).mean().item() if dice_tissue else 0.0
    avg_dice_n = torch.cat(dice_nuclei).mean().item() if dice_nuclei else 0.0
    return avg_loss, avg_dice_t, avg_dice_n


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: warmup_cosine_lr(epoch, config.warmup_epochs, config.epochs),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
        self.criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=config.log_dir)

        self.best_dice = 0.0
        self.start_epoch = 0

    def fit(self) -> float:
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.scaler, epoch, self.config, self.device,
            )
            self.scheduler.step()

            val_loss, val_dice_t, val_dice_n = validate(
                self.model, self.val_loader, self.criterion, self.config, self.device,
            )

            self._writer.add_scalar("Loss/train", train_loss, epoch)
            self._writer.add_scalar("Loss/val", val_loss, epoch)
            self._writer.add_scalar("Dice/tissue", val_dice_t, epoch)
            self._writer.add_scalar("Dice/nuclei", val_dice_n, epoch)
            self._writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            print(
                f"─── E{epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"dice_t={val_dice_t:.4f}  dice_n={val_dice_n:.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            monitor = val_dice_t + val_dice_n if val_dice_n > 0 else val_dice_t
            if monitor > self.best_dice:
                self.best_dice = monitor
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "best_dice": self.best_dice,
                    "config": self.config,
                }
                torch.save(
                    ckpt,
                    os.path.join(self.config.ckpt_dir, f"{self.config.model_type}_best.pth"),
                )
                print(f"  → Saved best (dice_t={val_dice_t:.4f}, dice_n={val_dice_n:.4f})")

            if (epoch + 1) % self.config.save_interval == 0:
                torch.save(
                    ckpt,
                    os.path.join(
                        self.config.ckpt_dir,
                        f"{self.config.model_type}_epoch{epoch:03d}.pth",
                    ),
                )

        self._writer.close()
        print(f"\nDone! Best monitor dice: {self.best_dice:.4f}")
        return self.best_dice

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_dice = ckpt["best_dice"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {self.start_epoch}, best_dice={self.best_dice:.4f}")
