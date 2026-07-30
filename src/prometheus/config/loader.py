"""Load experiment configuration from TOML, rejecting every unknown key.

Silently ignoring an unknown key is how a misspelled hyper-parameter becomes a run that
looks fine and trains with the default value. Every section and field is allow-listed.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from .schema import (
    DataConfig,
    EngineConfig,
    EvaluationConfig,
    LossConfig,
    OptimizerConfig,
    PathsConfig,
    PostprocessConfig,
    ProjectConfig,
    PrometheusModelConfig,
)

__all__ = ["load_project_config"]

_SECTIONS = frozenset(
    {
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
)
_EXPERIMENT_FIELDS = frozenset({"name", "seed"})
_CONTEXT_FIELDS = frozenset({"enabled"})

DataclassT = TypeVar("DataclassT")


def _reject_unknown(name: str, values: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")


def _strict(cls: type[DataclassT], values: dict[str, Any]) -> DataclassT:
    # `fields` is only typed for DataclassInstance; every caller passes a dataclass type.
    known = frozenset(field.name for field in fields(cls) if not field.name.startswith("_"))  # type: ignore[arg-type]
    _reject_unknown(cls.__name__, values, known)
    return cls(**values)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Parse and validate an experiment TOML file into a :class:`ProjectConfig`."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    _reject_unknown("configuration section", raw, _SECTIONS)
    experiment = raw.get("experiment", {})
    _reject_unknown("experiment", experiment, _EXPERIMENT_FIELDS)

    model_values = dict(raw.get("model", {}))
    context = model_values.pop("context", {})
    _reject_unknown("model.context", context, _CONTEXT_FIELDS)
    if "enabled" in context:
        model_values["context_enabled"] = context["enabled"]

    config = ProjectConfig(
        model=_strict(PrometheusModelConfig, model_values),
        data=_strict(DataConfig, raw.get("data", {})),
        optimizer=_strict(OptimizerConfig, raw.get("optimizer", {})),
        trainer=_strict(EngineConfig, raw.get("trainer", {})),
        loss=_strict(LossConfig, raw.get("loss", {})),
        evaluation=_strict(EvaluationConfig, raw.get("evaluation", {})),
        paths=_strict(PathsConfig, raw.get("paths", {})),
        postprocess=_strict(PostprocessConfig, raw.get("postprocess", {})),
        name=experiment.get("name", "prometheus_experiment"),
        seed=experiment.get("seed", 42),
    )
    config.validate()
    return config
