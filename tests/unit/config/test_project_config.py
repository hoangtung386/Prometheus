from pathlib import Path

import pytest

from prometheus.api import build_criterion
from prometheus.config import load_project_config


def test_refactored_config_loads() -> None:
    config = load_project_config("configs/experiment/baseline_multitask.toml")
    assert config.model.name == "prometheus_multitask_v1"
    assert config.model.num_nucleus_types == 10
    assert config.trainer.gradient_accumulation == 1


def test_configured_criterion_includes_gaussian_radius() -> None:
    config = load_project_config("configs/experiment/baseline_multitask.toml")
    config.loss.gaussian_radius = 5
    criterion = build_criterion(config)
    assert criterion.gaussian_radius == 5
    assert criterion.weights.tissue_ce == config.loss.tissue_ce


def test_refactored_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[model]\nunknown_magic = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown PrometheusModelConfig fields"):
        load_project_config(path)
