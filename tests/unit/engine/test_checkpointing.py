from pathlib import Path

import pytest

from prometheus.config import ProjectConfig, PrometheusModelConfig
from prometheus.engine import assert_checkpoint_compatible, load_engine_checkpoint, save_engine_checkpoint
from prometheus.models import PrometheusNet


def test_schema_v2_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ProjectConfig(
        model=PrometheusModelConfig(
            encoder_dims=[8, 16, 32, 64],
            encoder_depths=[1, 1, 1, 1],
            tissue_decoder_depths=[1, 1, 1],
        )
    )
    model = PrometheusNet(config.model)
    path = tmp_path / "model.ckpt"
    save_engine_checkpoint(path, model, config, 3, 17, {"nuclei/macro_f1_summed": 0.5})
    payload = load_engine_checkpoint(path)
    assert payload["schema_version"] == 2
    assert payload["architecture"] == "prometheus_multitask_v1"
    assert payload["epoch"] == 3
    assert payload["global_step"] == 17
    assert_checkpoint_compatible(payload, config)
    assert not (tmp_path / "model.ckpt.tmp").exists()


def test_checkpoint_rejects_different_model_config(tmp_path: Path) -> None:
    config = ProjectConfig(
        model=PrometheusModelConfig(
            encoder_dims=[8, 16, 32, 64],
            encoder_depths=[1, 1, 1, 1],
            tissue_decoder_depths=[1, 1, 1],
        )
    )
    path = tmp_path / "model.ckpt"
    save_engine_checkpoint(path, PrometheusNet(config.model), config, 0, 0, {})
    payload = load_engine_checkpoint(path)
    config.model.context_enabled = False
    with pytest.raises(ValueError, match="does not match"):
        assert_checkpoint_compatible(payload, config)
