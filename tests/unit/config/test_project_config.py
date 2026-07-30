from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.api import build_criterion
from prometheus.config import CHECKPOINT_METRICS, ProjectConfig, load_project_config

REFERENCE_CONFIG = "configs/experiment/baseline_multitask.toml"


def test_reference_config_loads_and_validates() -> None:
    config = load_project_config(REFERENCE_CONFIG)

    assert config.model.name == "prometheus_multitask_v1"
    assert config.model.num_nucleus_types == 10
    assert config.model.num_tissue_classes == 6
    assert config.trainer.gradient_accumulation == 1
    assert config.evaluation.checkpoint_metric in CHECKPOINT_METRICS


def test_configured_criterion_carries_the_loss_settings() -> None:
    config = load_project_config(REFERENCE_CONFIG)
    config.loss.gaussian_radius = 5
    criterion = build_criterion(config)

    assert criterion.gaussian_radius == 5
    assert criterion.weights.tissue_ce == config.loss.tissue_ce
    assert criterion.nuclei_class_weights is None, "evaluation loss must not depend on a fold's histogram"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[model]\nunknown_magic = true\n", "Unknown PrometheusModelConfig fields"),
        ("[nonsense]\nx = 1\n", "Unknown configuration section fields"),
        ("[experiment]\nauthor = 'me'\n", "Unknown experiment fields"),
        ("[model.context]\nenabled_typo = true\n", "Unknown model.context fields"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    # Silently ignoring a misspelled key produces a run that looks fine and trains with the
    # default value, which is far more expensive to find than a load-time failure.
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_project_config(path)


def test_checkpoint_metric_must_be_a_supported_metric() -> None:
    config = ProjectConfig()
    config.evaluation.checkpoint_metric = "tissue/dice_mean_fg"

    with pytest.raises(ValueError, match="checkpoint_metric"):
        config.validate()


def test_the_official_tissue_metric_is_selectable() -> None:
    config = ProjectConfig()
    config.evaluation.checkpoint_metric = "tissue/micro_dice"

    config.validate()


@pytest.mark.parametrize("power", [0.0, 1.5, -0.5])
def test_class_weight_power_is_bounded(power: float) -> None:
    config = ProjectConfig()
    config.loss.class_weight_power = power

    with pytest.raises(ValueError, match="class_weight_power"):
        config.validate()


def test_min_lr_may_not_exceed_the_learning_rate() -> None:
    config = ProjectConfig()
    config.trainer.min_lr = config.optimizer.lr * 10

    with pytest.raises(ValueError, match="min_lr"):
        config.validate()


def test_local_max_kernel_must_be_odd() -> None:
    config = ProjectConfig()
    config.postprocess.local_max_kernel = 4

    with pytest.raises(ValueError, match="local_max_kernel"):
        config.validate()
