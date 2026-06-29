"""Validated configuration schemas."""

from .loader import load_experiment_config
from .schemas import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)

DEFAULT_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "load_experiment_config",
]
