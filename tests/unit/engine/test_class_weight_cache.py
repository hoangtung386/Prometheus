"""Class-weight caching.

Computing the weights scans the whole training set, so the result is cached in the run
directory. That directory lives on Drive and outlives any config change, so the cache must be
keyed on everything that determines its contents. Keying it on file existence alone means a
changed weighting policy is silently ignored while the config claims it is active.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prometheus.config import EngineConfig, PathsConfig, ProjectConfig, PrometheusModelConfig
from prometheus.domain import ImageMeta, MultitaskBatch, NucleiTarget, TissueTarget
from prometheus.engine import PrometheusTrainer
from prometheus.losses import BACKGROUND_TISSUE_WEIGHT
from prometheus.models import PrometheusNet

CACHE_NAME = "class_weights.json"


def _batch() -> MultitaskBatch:
    mask = torch.zeros(1, 32, 32, dtype=torch.long)
    mask[0, :16, :] = 1  # tumor
    mask[0, 16:20, :] = 4  # necrosis, deliberately rarer
    return MultitaskBatch(
        images=torch.zeros(1, 3, 32, 32),
        tissue=TissueTarget(mask),
        nuclei=[
            NucleiTarget(
                centroids=torch.tensor([[8.0, 8.0]]),
                labels=torch.tensor([0]),
                boxes=torch.tensor([[6.0, 6.0, 10.0, 10.0]]),
            )
        ],
        metadata=[ImageMeta("sample", (32, 32), (32, 32), (32, 32), (1.0, 1.0), (0, 0))],
    )


def _config(tmp_path: Path, power: float = 0.5) -> ProjectConfig:
    config = ProjectConfig(
        model=PrometheusModelConfig(
            encoder_dims=[8, 16, 32, 64],
            encoder_depths=[1, 1, 1, 1],
            tissue_decoder_depths=[1, 1, 1],
        ),
        trainer=EngineConfig(epochs=1, batch_size=1, num_workers=0, amp=False, warmup_epochs=0),
        paths=PathsConfig(run_dir=str(tmp_path)),
    )
    config.loss.class_weighting = True
    config.loss.class_weight_power = power
    return config


def _trainer(config: ProjectConfig) -> PrometheusTrainer:
    batch = _batch()
    return PrometheusTrainer(PrometheusNet(config.model), [batch], [batch], config, torch.device("cpu"))


def test_weights_are_computed_and_cached_with_their_signature(tmp_path: Path) -> None:
    trainer = _trainer(_config(tmp_path))
    cached = json.loads((tmp_path / CACHE_NAME).read_text(encoding="utf-8"))

    assert cached["signature"]["power"] == 0.5
    # float32 storage, so compare approximately.
    assert cached["tissue"][0] == pytest.approx(BACKGROUND_TISSUE_WEIGHT)
    assert trainer.criterion.tissue.dice.class_weights is not None
    # Necrosis (index 4) is the rarest scored class here and must weigh more than tumour.
    assert cached["tissue"][4] > cached["tissue"][1]


def test_an_unchanged_policy_reuses_the_cache(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _trainer(_config(tmp_path))
    capsys.readouterr()
    _trainer(_config(tmp_path))

    output = capsys.readouterr().out
    assert "Computing class weights" not in output
    assert "stale" not in output


def test_a_changed_power_invalidates_the_cache(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _trainer(_config(tmp_path, power=1.0))
    first = json.loads((tmp_path / CACHE_NAME).read_text(encoding="utf-8"))
    capsys.readouterr()

    _trainer(_config(tmp_path, power=0.5))
    second = json.loads((tmp_path / CACHE_NAME).read_text(encoding="utf-8"))

    assert "stale" in capsys.readouterr().out
    assert second["signature"]["power"] == 0.5
    assert second["tissue"] != first["tissue"], "recomputing must actually change the weights"


def test_a_cache_without_a_signature_is_treated_as_stale(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # This is the shape written by the previous revision, which used plain inverse frequency
    # and gave the unscored background the largest weight of all.
    (tmp_path / CACHE_NAME).write_text(
        json.dumps({"tissue": [3.782, 0.009, 0.024, 0.311, 1.045, 0.828], "nuclei": [1.0] * 10}),
        encoding="utf-8",
    )
    trainer = _trainer(_config(tmp_path))

    assert "stale" in capsys.readouterr().out
    weights = trainer.criterion.tissue.dice.class_weights
    assert weights is not None
    assert weights[0] == pytest.approx(BACKGROUND_TISSUE_WEIGHT), "the legacy weight must not survive"


def test_disabling_class_weighting_yields_no_weights(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.loss.class_weighting = False
    trainer = _trainer(config)

    assert trainer.criterion.tissue.dice.class_weights is None
    assert trainer.criterion.nuclei_class_weights is None
    assert not (tmp_path / CACHE_NAME).exists()
