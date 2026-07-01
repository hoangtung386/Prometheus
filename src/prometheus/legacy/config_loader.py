"""TOML loader retained for legacy semantic experiment files."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from ..config.common import DataConfig, EvaluationConfig
from .config import ExperimentConfig, ModelConfig, TrainingConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as file_obj:
        raw = tomllib.load(file_obj)
    config = ExperimentConfig(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
    config.validate()
    return config
