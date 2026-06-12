from __future__ import annotations

import math
import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..data.puma_dataset import NUCLEI_CLASSES, TISSUE_CLASSES
from ..losses import MulticlassCombinedLoss
from ..metrics import SegmentationEvaluator


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    include_background: bool = False,
    ignore_absent: bool = True,
) -> torch.Tensor:
    num_classes = pred.shape[1]
    pred_idx = pred.argmax(dim=1)
    pred_onehot = F.one_hot(pred_idx, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    cardinality = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2 * intersection) / cardinality.clamp_min(eps)

    valid = torch.ones_like(dice, dtype=torch.bool)
    if not include_background and num_classes > 1:
        valid[:, 0] = False
    if ignore_absent:
        valid &= target_onehot.sum(dim=(2, 3)) > 0

    scores = []
    for sample_dice, sample_valid in zip(dice, valid):
        if sample_valid.any():
            scores.append(sample_dice[sample_valid].mean())
        else:
            scores.append(torch.tensor(float("nan"), device=pred.device))
    return torch.stack(scores)


def per_class_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = pred.shape[1]
    pred_idx = pred.argmax(dim=1)
    pred_onehot = F.one_hot(pred_idx, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    cardinality = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2. * intersection + eps) / (cardinality + eps)
    return dice  # (B, C)


def warmup_cosine_lr(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> float:
    if epoch < warmup_epochs:
        return max(min_lr_ratio, epoch / max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


CriterionLike = Union[nn.Module, dict[str, nn.Module]]


def _criterion_for(criterion: CriterionLike, modality: str) -> nn.Module:
    if isinstance(criterion, dict):
        return criterion[modality]
    return criterion


def _loss_components(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(criterion, "components"):
        ce, dice = criterion.components(logits, targets)
        return criterion.ce_weight * ce + criterion.dice_weight * dice, ce, dice
    loss = criterion(logits, targets)
    nan = torch.tensor(float("nan"), device=logits.device)
    return loss, nan, nan


def compute_class_weights(
    loader: DataLoader,
    modality: str,
    num_classes: int,
    power: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, targets in loader:
        mask = targets[modality]
        counts += torch.bincount(mask.reshape(-1), minlength=num_classes).double()

    present = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32)
    if present.any():
        freq = counts[present] / counts[present].sum().clamp_min(eps)
        inv = (1.0 / freq.clamp_min(eps)).pow(power)
        inv = inv / inv.mean().clamp_min(eps)
        weights[present] = inv.float()
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CriterionLike,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: TrainingConfig,
    device: torch.device,
    return_details: bool = False,
) -> float | dict[str, float]:
    model.train()
    total_loss = 0.0
    totals = {
        "loss": 0.0,
        "tissue_ce": 0.0,
        "tissue_dice": 0.0,
        "nuclei_ce": 0.0,
        "nuclei_dice": 0.0,
        "moe": 0.0,
        "grad_norm": 0.0,
    }
    n = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        t_mask = targets["tissue"].to(device, non_blocking=True)
        n_mask = targets["nuclei"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=config.amp):
            if config.model_type == "UNetTissue":
                logits = model(images)
                loss, t_ce, t_dice = _loss_components(
                    _criterion_for(criterion, "tissue"), logits, t_mask,
                )
                n_ce = n_dice = moe_loss = torch.tensor(float("nan"), device=device)
            else:
                pred_t, pred_n, moe_loss = model(images)
                t_loss, t_ce, t_dice = _loss_components(
                    _criterion_for(criterion, "tissue"), pred_t, t_mask,
                )
                n_loss, n_ce, n_dice = _loss_components(
                    _criterion_for(criterion, "nuclei"), pred_n, n_mask,
                )
                loss = (
                    t_loss
                    + n_loss
                    + config.moe_loss_weight * moe_loss
                )

        scaler.scale(loss).backward()
        grad_norm = torch.tensor(0.0, device=device)
        if config.gradient_clip_norm and config.gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        totals["loss"] += loss.item()
        for key, value in [
            ("tissue_ce", t_ce),
            ("tissue_dice", t_dice),
            ("nuclei_ce", n_ce),
            ("nuclei_dice", n_dice),
            ("moe", moe_loss),
            ("grad_norm", grad_norm),
        ]:
            if torch.isfinite(value):
                totals[key] += value.item()

        if batch_idx % config.log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"E{epoch:03d} B{batch_idx:04d}/{n}  loss={loss.item():.4f}  lr={lr_now:.2e}")

    if not return_details:
        return total_loss / n
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CriterionLike,
    config: TrainingConfig,
    device: torch.device,
    eval_t: Optional[SegmentationEvaluator] = None,
    eval_n: Optional[SegmentationEvaluator] = None,
) -> tuple[float, float, float]:
    """Run evaluation loop.

    Returns (avg_loss, avg_dice_tissue, avg_dice_nuclei).
    When eval_t / eval_n are provided they accumulate per-class statistics.
    """
    model.eval()
    total_loss = 0.0
    dice_tissue, dice_nuclei = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        t_mask = targets["tissue"].to(device, non_blocking=True)
        n_mask = targets["nuclei"].to(device, non_blocking=True)

        if config.model_type == "UNetTissue":
            logits = model(images)
            loss = _criterion_for(criterion, "tissue")(logits, t_mask)
            dice_tissue.append(dice_score(logits, t_mask))
            if eval_t is not None:
                eval_t.update(logits, t_mask)
        else:
            pred_t, pred_n, _ = model(images)
            loss = (
                _criterion_for(criterion, "tissue")(pred_t, t_mask)
                + _criterion_for(criterion, "nuclei")(pred_n, n_mask)
            )
            dice_tissue.append(dice_score(pred_t, t_mask))
            dice_nuclei.append(dice_score(pred_n, n_mask))
            if eval_t is not None:
                eval_t.update(pred_t, t_mask)
            if eval_n is not None:
                eval_n.update(pred_n, n_mask)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    avg_dice_t = torch.nanmean(torch.cat(dice_tissue)).item() if dice_tissue else 0.0
    avg_dice_n = torch.nanmean(torch.cat(dice_nuclei)).item() if dice_nuclei else 0.0
    if math.isnan(avg_dice_t):
        avg_dice_t = 0.0
    if math.isnan(avg_dice_n):
        avg_dice_n = 0.0
    return avg_loss, avg_dice_t, avg_dice_n


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        test_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader or val_loader
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

        min_lr_ratio = config.scheduler_min_lr / config.lr if config.lr > 0 else 0.0
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: warmup_cosine_lr(
                epoch, config.warmup_epochs, config.epochs, min_lr_ratio,
            ),
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
        tissue_weights = None
        nuclei_weights = None
        if config.use_class_weights:
            tissue_weights = compute_class_weights(
                train_loader, "tissue", config.num_tissue_classes, config.class_weight_power,
            )
            if config.model_type != "UNetTissue":
                nuclei_weights = compute_class_weights(
                    train_loader, "nuclei", config.num_nuclei_classes, config.class_weight_power,
                )
            print(f"Class weights/tissue: {[round(v, 4) for v in tissue_weights.tolist()]}")
            if nuclei_weights is not None:
                print(f"Class weights/nuclei: {[round(v, 4) for v in nuclei_weights.tolist()]}")

        self.criterion: CriterionLike = {
            "tissue": MulticlassCombinedLoss(
                ce_weight=1.0, dice_weight=1.0, class_weights=tissue_weights,
            ).to(self.device),
            "nuclei": MulticlassCombinedLoss(
                ce_weight=1.0, dice_weight=1.0, class_weights=nuclei_weights,
            ).to(self.device),
        }

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=config.log_dir)
        except ModuleNotFoundError:
            print("Warning: tensorboard not installed, metrics will not be logged")
            self._writer = None

        self.best_dice = 0.0
        self.best_tissue_dice = 0.0
        self.best_nuclei_dice = 0.0
        self.start_epoch = 0
        self._epochs_without_improvement = 0

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_dice": self.best_dice,
            "best_tissue_dice": self.best_tissue_dice,
            "best_nuclei_dice": self.best_nuclei_dice,
            "config": self.config,
        }

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        torch.save(
            self._checkpoint_payload(epoch),
            os.path.join(self.config.ckpt_dir, name),
        )

    def fit(self) -> float:
        for epoch in range(self.start_epoch, self.config.epochs):
            if hasattr(self.model, "set_tissue_context_enabled"):
                self.model.set_tissue_context_enabled(
                    epoch >= self.config.tissue_context_warmup_epochs,
                )
            train_metrics = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.scaler, epoch, self.config, self.device,
                return_details=True,
            )
            train_loss = train_metrics["loss"]
            self.scheduler.step()

            eval_t = SegmentationEvaluator(
                len(TISSUE_CLASSES), TISSUE_CLASSES,
            )
            eval_n = SegmentationEvaluator(
                len(NUCLEI_CLASSES), NUCLEI_CLASSES,
            ) if self.config.model_type != "UNetTissue" else None

            test_loss, test_dice_t, test_dice_n = validate(
                self.model, self.test_loader, self.criterion, self.config, self.device,
                eval_t=eval_t, eval_n=eval_n,
            )
            log_t = eval_t.log_dict("tissue")
            log_n = eval_n.log_dict("nuclei") if eval_n is not None else {}
            test_dice_t = log_t["tissue/Dice/mean_present_fg"]
            test_dice_n = log_n.get("nuclei/Dice/mean_present_fg", 0.0)

            if self._writer is not None:
                self._writer.add_scalar("Loss/train", train_loss, epoch)
                self._writer.add_scalar("Loss/test", test_loss, epoch)
                self._writer.add_scalar("Dice/tissue", test_dice_t, epoch)
                self._writer.add_scalar("Dice/nuclei", test_dice_n, epoch)
                self._writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
                for name, val in train_metrics.items():
                    self._writer.add_scalar(f"Loss/train_components/{name}", val, epoch)
                for name, val in log_t.items():
                    self._writer.add_scalar(name, val, epoch)
                if eval_n is not None:
                    for name, val in log_n.items():
                        self._writer.add_scalar(name, val, epoch)
                self._writer.add_scalar("Best/tissue", self.best_tissue_dice, epoch)
                self._writer.add_scalar("Best/nuclei", self.best_nuclei_dice, epoch)

            print(f"\n{'='*70}")
            print(f"  Epoch {epoch:03d}  |  Train loss: {train_loss:.4f}  |  Test loss: {test_loss:.4f}  |  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Mean Dice — tissue: {test_dice_t:.4f}  nuclei: {test_dice_n:.4f}")
            print(eval_t.summary("Tissue"))
            if eval_n is not None:
                print(eval_n.summary("Nuclei"))
            print(f"{'='*70}\n")

            monitor = test_dice_t + test_dice_n if test_dice_n > 0 else test_dice_t
            if self.config.early_stopping_monitor == "tissue":
                monitor = test_dice_t
            elif self.config.early_stopping_monitor == "nuclei":
                monitor = test_dice_n
            elif self.config.early_stopping_monitor != "combined":
                raise ValueError(
                    "early_stopping_monitor must be one of: combined, tissue, nuclei"
                )

            if test_dice_t > self.best_tissue_dice:
                self.best_tissue_dice = test_dice_t
                self._save_checkpoint(epoch, f"{self.config.model_type}_best_tissue.pth")
                print(f"  + Saved best tissue (dice_t={test_dice_t:.4f})")
            if test_dice_n > self.best_nuclei_dice:
                self.best_nuclei_dice = test_dice_n
                self._save_checkpoint(epoch, f"{self.config.model_type}_best_nuclei.pth")
                if self.config.model_type != "UNetTissue":
                    print(f"  + Saved best nuclei (dice_n={test_dice_n:.4f})")

            if monitor > self.best_dice:
                self.best_dice = monitor
                self._epochs_without_improvement = 0
                self._save_checkpoint(epoch, f"{self.config.model_type}_best.pth")
                print(f"  + Saved best (dice_t={test_dice_t:.4f}, dice_n={test_dice_n:.4f})")
            else:
                self._epochs_without_improvement += 1

            self._save_checkpoint(epoch, f"{self.config.model_type}_last.pth")

            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch, f"{self.config.model_type}_epoch{epoch:03d}.pth")

            if (
                self.config.early_stopping_patience is not None
                and self._epochs_without_improvement >= self.config.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch:03d} after "
                    f"{self._epochs_without_improvement} epochs without improvement"
                )
                break

        if self._writer is not None:
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
        self.best_tissue_dice = ckpt.get("best_tissue_dice", self.best_dice)
        self.best_nuclei_dice = ckpt.get("best_nuclei_dice", 0.0)
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {self.start_epoch}, best_dice={self.best_dice:.4f}")
