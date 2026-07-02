"""Small training engine dedicated to typed Prometheus contracts."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import ProjectConfig
from ..losses import LossWeights, PrometheusMultitaskLoss
from .checkpointing import assert_checkpoint_compatible, load_engine_checkpoint, save_engine_checkpoint
from .evaluator import evaluate_multitask


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrometheusTrainer:
    def __init__(self, model, train_loader, validation_loader, config: ProjectConfig, device=None) -> None:
        config.validate()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.run_dir = Path(config.paths.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "resolved_config.json").write_text(
            json.dumps(asdict(config), indent=2),
            encoding="utf-8",
        )
        _seed_everything(config.seed)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
        )
        minimum_ratio = config.trainer.min_lr / config.optimizer.lr
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda(minimum_ratio))
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.trainer.amp and self.device.type == "cuda")
        self.criterion = PrometheusMultitaskLoss(
            config.model.num_nucleus_types,
            config.model.nuclei_feature_stride,
            LossWeights(**{name: getattr(config.loss, name) for name in LossWeights.__dataclass_fields__}),
            gaussian_radius=config.loss.gaussian_radius,
        )
        self.ema_decay = config.trainer.ema_decay
        self.ema_model = None
        if self.ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            for parameter in self.ema_model.parameters():
                parameter.requires_grad_(False)
        self.start_epoch = 0
        self.global_step = 0
        self.best_primary = float("-inf")
        self.best_tissue = float("-inf")
        self.history: list[dict] = []

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.ema_decay
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters(), strict=True
        ):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
            ema_buffer.copy_(buffer)

    def _lr_lambda(self, minimum_ratio: float):
        warmup, epochs = self.config.trainer.warmup_epochs, self.config.trainer.epochs

        def schedule(epoch: int) -> float:
            if epoch < warmup:
                return max(minimum_ratio, (epoch + 1) / max(warmup, 1))
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return schedule

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        accumulation = self.config.trainer.gradient_accumulation
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(self.train_loader):
            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                output = self.model(batch.images)
                losses = self.criterion(output, batch)
                scaled_loss = losses["total"] / accumulation
            self.scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(self.train_loader)
            if should_step:
                self.scaler.unscale_(self.optimizer)
                clip = self.config.trainer.gradient_clip_norm
                if clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    totals["diagnostics/grad_norm"] = totals.get("diagnostics/grad_norm", 0.0) + float(grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.ema_model is not None:
                    self._update_ema()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
            if batch_index % self.config.trainer.log_interval == 0:
                print(
                    f"Epoch {epoch:03d} batch {batch_index:04d}/{len(self.train_loader):04d} "
                    f"loss={losses['total'].detach().item():.4f}"
                )
        return {name: value / max(len(self.train_loader), 1) for name, value in totals.items()}

    def fit(self, resume_from: str | Path | None = None) -> dict[str, float]:
        if resume_from is not None:
            self.resume(resume_from)
        elif (self.run_dir / "metrics.jsonl").exists():
            history_path = self.run_dir / "metrics.jsonl"
            loaded = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if loaded:
                self.history = loaded
                self.start_epoch = loaded[-1]["epoch"] + 1
                self.best_primary = loaded[-1].get(self.config.evaluation.checkpoint_metric, self.best_primary)
        last_metrics = {}
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_metrics = self.train_epoch(epoch)
            evaluation_model = self.ema_model if self.ema_model is not None else self.model
            evaluation = evaluate_multitask(
                evaluation_model,
                self.validation_loader,
                self.criterion,
                self.device,
                self.config.model.nuclei_feature_stride,
                self.config.evaluation.nuclei_radius_px,
                self.config.postprocess.confidence_threshold,
                self.config.postprocess.max_detections,
                self.config.postprocess.local_max_kernel,
            )
            self.scheduler.step()
            last_metrics = {
                **{f"train/{name}": value for name, value in train_metrics.items()},
                "validation/loss": evaluation.loss,
                "tissue/dice_mean_fg": evaluation.tissue_dice,
                "nuclei/macro_f1_summed": evaluation.nuclei_macro_f1,
            }
            primary_metric = last_metrics[self.config.evaluation.checkpoint_metric]
            improved_primary = primary_metric > self.best_primary
            improved_tissue = evaluation.tissue_dice > self.best_tissue
            if improved_primary:
                self.best_primary = primary_metric
            if improved_tissue:
                self.best_tissue = evaluation.tissue_dice
            self._save(epoch, "last.ckpt", last_metrics)
            if improved_primary:
                self._save(epoch, "best_primary.ckpt", last_metrics)
            if improved_tissue:
                self._save(epoch, "best_tissue.ckpt", last_metrics)
            (self.run_dir / "metrics.json").write_text(json.dumps(last_metrics, indent=2), encoding="utf-8")
            history_record = {
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                **last_metrics,
            }
            self.history.append(history_record)
            with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(history_record) + "\n")
            peak_memory = (
                f" peak_vram={torch.cuda.max_memory_allocated(self.device) / 2**30:.2f}GiB"
                if self.device.type == "cuda"
                else ""
            )
            print(
                f"Epoch {epoch:03d} loss={evaluation.loss:.4f} "
                f"tissue_dice={evaluation.tissue_dice:.4f} nuclei_f1={evaluation.nuclei_macro_f1:.4f}"
                f"{peak_memory}"
            )
        return last_metrics

    def _save(self, epoch: int, name: str, metrics: dict[str, float]) -> None:
        checkpoint_metrics = {
            **metrics,
            "engine/best_primary": self.best_primary,
            "engine/best_tissue": self.best_tissue,
        }
        save_engine_checkpoint(
            self.run_dir / name,
            self.model,
            self.config,
            epoch,
            self.global_step,
            checkpoint_metrics,
            self.optimizer,
            self.scheduler,
            self.scaler,
            ema_state=self.ema_model.state_dict() if self.ema_model is not None else None,
        )

    def resume(self, path: str | Path) -> None:
        checkpoint = load_engine_checkpoint(path, self.device)
        assert_checkpoint_compatible(checkpoint, self.config)
        self.model.load_state_dict(checkpoint["model_state"])
        if self.ema_model is not None and checkpoint.get("ema_state") is not None:
            self.ema_model.load_state_dict(checkpoint["ema_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint["scaler_state"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        metrics = checkpoint.get("metrics", {})
        self.best_primary = float(metrics.get("engine/best_primary", self.best_primary))
        self.best_tissue = float(metrics.get("engine/best_tissue", self.best_tissue))
        rng = checkpoint.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if self.device.type == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
