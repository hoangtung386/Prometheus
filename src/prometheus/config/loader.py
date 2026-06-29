"""Load experiment configuration from standard-library TOML files."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from .schemas import DataConfig, EvaluationConfig, ExperimentConfig, ModelConfig, TrainingConfig


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
