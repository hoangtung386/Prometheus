"""Training loop for the typed Prometheus contracts."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import ProjectConfig
from ..data.puma import TISSUE_PAINT_PRIORITY
from ..losses import LossWeights, PrometheusMultitaskLoss, compute_class_weights
from ..models import PrometheusNet
from .checkpointing import assert_checkpoint_compatible, load_engine_checkpoint, save_engine_checkpoint
from .ema import WeightEma
from .run_log import RunLog
from .schedule import build_optimizer, warmup_cosine_lambda
from .validation import evaluate_multitask

__all__ = ["PrometheusTrainer"]

_CLASS_WEIGHT_CACHE = "class_weights.json"
_CLASS_WEIGHT_CACHE_SCHEMA = 2
_HISTORY_FILE = "metrics.jsonl"
_LOG_FILE = "train.log"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch, including all CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrometheusTrainer:
    """Train :class:`~prometheus.models.PrometheusNet` and checkpoint per task.

    Two checkpoints are maintained because the challenge ranks the two tasks
    independently, so the best tissue epoch and the best nuclei epoch are usually not the
    same epoch:

    * ``best_primary.ckpt`` — best ``config.evaluation.checkpoint_metric``;
    * ``best_tissue.ckpt`` — best official tissue micro Dice.

    ``last.ckpt`` supports exact resume, which the supported Colab workstation needs
    because runtimes disconnect.
    """

    def __init__(
        self,
        model: PrometheusNet,
        train_loader,
        validation_loader,
        config: ProjectConfig,
        device: torch.device | None = None,
    ) -> None:
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
        self.log = RunLog(self.run_dir / _LOG_FILE)
        self.log.session(f"run {config.name} | device {self.device} | seed {config.seed}")
        seed_everything(config.seed)

        self.optimizer = build_optimizer(
            model,
            lr=config.optimizer.lr,
            backbone_lr_multiplier=config.optimizer.backbone_lr_multiplier,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            warmup_cosine_lambda(
                config.trainer.warmup_epochs,
                config.trainer.epochs,
                config.trainer.min_lr / config.optimizer.lr,
            ),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.trainer.amp and self.device.type == "cuda")
        tissue_weights, nuclei_weights = self._resolve_class_weights()
        self.criterion = PrometheusMultitaskLoss(
            config.model.num_nucleus_types,
            config.model.nuclei_feature_stride,
            LossWeights(**{name: getattr(config.loss, name) for name in LossWeights.field_names()}),
            gaussian_radius=config.loss.gaussian_radius,
            tissue_class_weights=tissue_weights,
            nuclei_class_weights=nuclei_weights,
        ).to(self.device)
        self.ema = WeightEma(self.model, config.trainer.ema_decay) if config.trainer.ema_decay > 0.0 else None

        self.start_epoch = 0
        self.global_step = 0
        self.best_primary = float("-inf")
        self.best_tissue = float("-inf")
        self.history: list[dict] = []

    # --- setup helpers ---------------------------------------------------------------

    def _class_weight_signature(self) -> dict[str, object]:
        """Everything that changes the computed weights, so a stale cache cannot be reused.

        Keying the cache on file existence alone is not enough. Both the weighting policy
        (``class_weight_power``) and the *masks the counts are taken from* (the tissue paint
        priority) can change between runs, and a run directory on Drive outlives either
        change. A silently reused cache would apply the old weights while the config claims
        the new ones.
        """
        return {
            "schema": _CLASS_WEIGHT_CACHE_SCHEMA,
            "power": self.config.loss.class_weight_power,
            "num_tissue_classes": self.config.model.num_tissue_classes,
            "num_nucleus_types": self.config.model.num_nucleus_types,
            "tissue_paint_priority": [item.value for item in TISSUE_PAINT_PRIORITY],
        }

    def _resolve_class_weights(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(tissue, nuclei)`` class weights, computing them at most once per policy.

        Computing them scans the whole training loader, so the result is cached in the run
        directory. The cache is invalidated whenever anything in
        :meth:`_class_weight_signature` changes.
        """
        if not self.config.loss.class_weighting:
            return None, None

        cache_path = self.run_dir / _CLASS_WEIGHT_CACHE
        signature = self._class_weight_signature()
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return torch.tensor(cached["tissue"]), torch.tensor(cached["nuclei"])
            self.log(f"Class-weight cache is stale (policy or masks changed); recomputing: {cache_path}")

        self.log("Computing class weights (one pass over the training data)...")
        tissue_weights, nuclei_weights = compute_class_weights(
            self.train_loader,
            self.config.model.num_tissue_classes,
            self.config.model.num_nucleus_types,
            power=self.config.loss.class_weight_power,
        )
        cache_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "tissue": tissue_weights.tolist(),
                    "nuclei": nuclei_weights.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.log(f"  tissue weights: {[round(value, 3) for value in tissue_weights.tolist()]}")
        self.log(f"  nuclei weights: {[round(value, 3) for value in nuclei_weights.tolist()]}")
        return tissue_weights, nuclei_weights

    # --- training --------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one epoch and return the mean of every loss component."""
        self.model.train()
        totals: dict[str, float] = {}
        accumulation = self.config.trainer.gradient_accumulation
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(self.train_loader):
            batch = raw_batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                losses = self.criterion(self.model(batch.images), batch)
                scaled_loss = losses["total"] / accumulation
            self.scaler.scale(scaled_loss).backward()

            last_batch = batch_index + 1 == len(self.train_loader)
            if (batch_index + 1) % accumulation == 0 or last_batch:
                self.scaler.unscale_(self.optimizer)
                clip = self.config.trainer.gradient_clip_norm
                if clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    totals["diagnostics/grad_norm"] = totals.get("diagnostics/grad_norm", 0.0) + float(grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)

            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            if batch_index % self.config.trainer.log_interval == 0:
                self.log(
                    f"Epoch {epoch:03d} batch {batch_index:04d}/{len(self.train_loader):04d} "
                    f"loss={float(losses['total'].detach()):.4f}"
                )
        return {name: value / max(len(self.train_loader), 1) for name, value in totals.items()}

    def fit(self, resume_from: str | Path | None = None) -> dict[str, float]:
        """Train to ``config.trainer.epochs``, resuming from ``resume_from`` when given."""
        if resume_from is not None:
            self.resume(resume_from)
        else:
            self._restore_history()

        last_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_metrics = self.train_epoch(epoch)
            evaluation = evaluate_multitask(
                self.ema.model if self.ema is not None else self.model,
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
                **evaluation.metrics,
                "validation/loss": evaluation.loss,
                "tissue/micro_dice": evaluation.tissue_micro_dice,
                "nuclei/macro_f1_summed": evaluation.nuclei_macro_f1,
            }
            self._checkpoint(epoch, last_metrics, evaluation.tissue_micro_dice)
            self._record(epoch, last_metrics)
            self.log(self._epoch_summary(epoch, evaluation))
        return last_metrics

    # --- bookkeeping -----------------------------------------------------------------

    def _epoch_summary(self, epoch: int, evaluation) -> str:
        per_class = evaluation.metrics
        rare = " ".join(
            f"{name}={per_class.get(f'tissue/micro_dice/{name}', 0.0):.3f}" for name in ("necrosis", "blood_vessel")
        )
        memory = (
            f" peak_vram={torch.cuda.max_memory_allocated(self.device) / 2**30:.2f}GiB"
            if self.device.type == "cuda"
            else ""
        )
        return (
            f"Epoch {epoch:03d} loss={evaluation.loss:.4f} "
            f"tissue_micro_dice={evaluation.tissue_micro_dice:.4f} ({rare}) "
            f"nuclei_f1={evaluation.nuclei_macro_f1:.4f}{memory}"
        )

    def _checkpoint(self, epoch: int, metrics: dict[str, float], tissue_micro_dice: float) -> None:
        primary = metrics[self.config.evaluation.checkpoint_metric]
        improved_primary = primary > self.best_primary
        improved_tissue = tissue_micro_dice > self.best_tissue
        self.best_primary = max(self.best_primary, primary)
        self.best_tissue = max(self.best_tissue, tissue_micro_dice)
        self._save(epoch, "last.ckpt", metrics)
        if improved_primary:
            self._save(epoch, "best_primary.ckpt", metrics)
        if improved_tissue:
            self._save(epoch, "best_tissue.ckpt", metrics)

    def _record(self, epoch: int, metrics: dict[str, float]) -> None:
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        record = {
            "epoch": epoch,
            "global_step": self.global_step,
            "learning_rate": self.optimizer.param_groups[-1]["lr"],
            **metrics,
        }
        self.history.append(record)
        with (self.run_dir / _HISTORY_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _restore_history(self) -> None:
        """Continue an interrupted run from its metrics history when no checkpoint is given."""
        history_path = self.run_dir / _HISTORY_FILE
        if not history_path.exists():
            return
        loaded = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not loaded:
            return
        self.history = loaded
        self.start_epoch = loaded[-1]["epoch"] + 1
        self.best_primary = loaded[-1].get(self.config.evaluation.checkpoint_metric, self.best_primary)

    def _save(self, epoch: int, name: str, metrics: dict[str, float]) -> None:
        save_engine_checkpoint(
            self.run_dir / name,
            self.model,
            self.config,
            epoch,
            self.global_step,
            {**metrics, "engine/best_primary": self.best_primary, "engine/best_tissue": self.best_tissue},
            self.optimizer,
            self.scheduler,
            self.scaler,
            ema_state=self.ema.state_dict() if self.ema is not None else None,
        )

    def resume(self, path: str | Path) -> None:
        """Restore weights, optimizer, schedule, scaler and RNG state for an exact resume."""
        checkpoint = load_engine_checkpoint(path, self.device)
        assert_checkpoint_compatible(checkpoint, self.config)
        self.model.load_state_dict(checkpoint["model_state"])
        if self.ema is not None and checkpoint.get("ema_state") is not None:
            self.ema.load_state_dict(checkpoint["ema_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint["scaler_state"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        metrics = checkpoint.get("metrics", {})
        self.best_primary = float(metrics.get("engine/best_primary", self.best_primary))
        self.best_tissue = float(metrics.get("engine/best_tissue", self.best_tissue))
        self._restore_rng(checkpoint.get("rng_state"))

    def _restore_rng(self, rng: dict | None) -> None:
        if not rng:
            return
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        # map_location may have moved the RNG tensors onto the GPU, but set_rng_state
        # requires CPU ByteTensors, so pull them back before restoring.
        torch.set_rng_state(rng["torch"].cpu())
        if self.device.type == "cuda" and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
