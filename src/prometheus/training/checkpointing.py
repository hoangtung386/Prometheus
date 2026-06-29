"""Versioned checkpoint persistence for training and inference."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from .state import TrainState

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_name: str,
    config,
    train_state: TrainState,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_name": model_name,
        "model_state": model.state_dict(),
        "train_state": train_state.to_dict(),
        "config": asdict(config) if is_dataclass(config) else config,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, destination)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "schema_version" not in payload:
        if "model_state_dict" in payload:
            return {
                "schema_version": 0,
                "model_name": payload.get("model_type"),
                "model_state": payload["model_state_dict"],
                "optimizer_state": payload.get("optimizer_state_dict"),
                "scheduler_state": payload.get("scheduler_state_dict"),
                "scaler_state": payload.get("scaler_state_dict"),
                "train_state": {
                    "epoch": payload.get("epoch", -1),
                    "global_step": payload.get("global_step", 0),
                    "best_metric": payload.get("best_dice", float("-inf")),
                    "best_tissue_metric": payload.get("best_tissue_dice", float("-inf")),
                    "best_nuclei_metric": payload.get("best_nuclei_dice", float("-inf")),
                    "early_stopping_counter": 0,
                },
                "config": payload.get("config"),
                "legacy": True,
            }
        return {"schema_version": 0, "model_state": payload, "legacy": True}
    if payload["schema_version"] > CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {payload['schema_version']}")
    return payload
