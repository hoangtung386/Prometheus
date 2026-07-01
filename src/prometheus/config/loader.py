"""Load experiment configuration from standard-library TOML files."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from .common import DataConfig, EvaluationConfig
from .project import (
    EngineConfig,
    LossConfig,
    OptimizerConfig,
    PathsConfig,
    PostprocessConfig,
    ProjectConfig,
    PrometheusModelConfig,
)


def _strict_dataclass(cls, values: dict):
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load the refactored config format and reject every unknown key."""
    with Path(path).open("rb") as file_obj:
        raw = tomllib.load(file_obj)
    allowed_sections = {
        "experiment",
        "model",
        "data",
        "optimizer",
        "trainer",
        "loss",
        "evaluation",
        "paths",
        "postprocess",
    }
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown_sections)}")
    experiment = raw.get("experiment", {})
    unknown_experiment = set(experiment) - {"name", "seed"}
    if unknown_experiment:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown_experiment)}")
    model_values = dict(raw.get("model", {}))
    context = model_values.pop("context", {})
    if set(context) - {"enabled"}:
        raise ValueError(f"Unknown model.context fields: {sorted(set(context) - {'enabled'})}")
    if "enabled" in context:
        model_values["context_enabled"] = context["enabled"]
    config = ProjectConfig(
        model=_strict_dataclass(PrometheusModelConfig, model_values),
        data=_strict_dataclass(DataConfig, raw.get("data", {})),
        optimizer=_strict_dataclass(OptimizerConfig, raw.get("optimizer", {})),
        trainer=_strict_dataclass(EngineConfig, raw.get("trainer", {})),
        loss=_strict_dataclass(LossConfig, raw.get("loss", {})),
        evaluation=(
            _strict_dataclass(EvaluationConfig, raw["evaluation"])
            if "evaluation" in raw
            else EvaluationConfig(checkpoint_metric="nuclei/macro_f1_summed")
        ),
        paths=_strict_dataclass(PathsConfig, raw.get("paths", {})),
        postprocess=_strict_dataclass(PostprocessConfig, raw.get("postprocess", {})),
        name=experiment.get("name", "prometheus_experiment"),
        seed=experiment.get("seed", 42),
    )
    config.validate()
    return config
