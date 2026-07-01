"""Architecture-aware checkpoint schema for the refactored engine."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import ProjectConfig

CHECKPOINT_SCHEMA_VERSION = 2


def save_engine_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: ProjectConfig,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": "prometheus_multitask_v1",
        "architecture_version": getattr(model, "architecture_version", 1),
        "model_state": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_engine_checkpoint(path: str | Path, map_location="cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Expected a Prometheus checkpoint with schema version 2")
    if payload.get("architecture") != "prometheus_multitask_v1":
        raise ValueError(f"Unsupported architecture: {payload.get('architecture')}")
    return payload


def assert_checkpoint_compatible(payload: dict, config: ProjectConfig) -> None:
    checkpoint_model = payload.get("config", {}).get("model")
    current_model = asdict(config.model)
    if checkpoint_model != current_model:
        raise ValueError(
            "Checkpoint model configuration does not match the requested model. "
            f"checkpoint={checkpoint_model}, requested={current_model}"
        )
