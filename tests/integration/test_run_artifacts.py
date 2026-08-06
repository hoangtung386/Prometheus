"""End-to-end proof that a training run leaves a usable model and a readable log on disk.

This runs the exact path the notebook's section 4 takes -- real dataset discovery, real
GeoJSON parsing, real rasterization, real dataloaders, real trainer -- on a tiny synthetic
PUMA root, and then checks what actually survives. It exists because "did it train?" and "can
I still tell what it did tomorrow?" are different questions, and on Colab the second one is
the one that bites: the runtime disconnects and every printed line is gone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from prometheus.api import build_datamodule, build_model, build_trainer
from prometheus.config import (
    DataConfig,
    EngineConfig,
    EvaluationConfig,
    PathsConfig,
    ProjectConfig,
    PrometheusModelConfig,
)
from prometheus.engine import load_engine_checkpoint, select_inference_state

EPOCHS = 2
IMAGE_SIZE = 64
CHECKPOINTS = ("last.ckpt", "best_primary.ckpt", "best_tissue.ckpt")
SIDECARS = ("resolved_config.json", "metrics.json", "metrics.jsonl", "train.log", "class_weights.json")


def _square(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _write_dataset(root: Path, count: int = 6) -> None:
    """A minimal but structurally faithful PUMA root: nested tissue, holes, several classes."""
    for name in ("images", "geojson_tissue", "geojson_nuclei"):
        (root / name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for index in range(count):
        site = "primary" if index % 2 else "metastatic"
        sample_id = f"{site}_roi_{index:03d}"
        tifffile.imwrite(
            root / "images" / f"{sample_id}.tif",
            rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        )
        tissue = {
            "type": "FeatureCollection",
            "features": [
                # Nested and listed first, so the paint priority is genuinely exercised.
                {
                    "properties": {"label": "tissue_necrosis"},
                    "geometry": {"type": "Polygon", "coordinates": [_square(10, 10, 20, 20)]},
                },
                {
                    "properties": {"label": "tissue_tumor"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_square(0, 0, 40, 40), _square(10, 10, 20, 20)],
                    },
                },
                {
                    "properties": {"label": "tissue_stroma"},
                    "geometry": {"type": "Polygon", "coordinates": [_square(40, 40, 60, 60)]},
                },
            ],
        }
        (root / "geojson_tissue" / f"{sample_id}_tissue.geojson").write_text(json.dumps(tissue), encoding="utf-8")
        nuclei = {
            "type": "FeatureCollection",
            "features": [
                {
                    "properties": {"classification": {"name": f"nuclei_{label}"}},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_square(5 + 12 * position, 5, 11 + 12 * position, 11)],
                    },
                }
                for position, label in enumerate(("tumor", "lymphocyte", "endothelium"))
            ],
        }
        (root / "geojson_nuclei" / f"{sample_id}_nuclei.geojson").write_text(json.dumps(nuclei), encoding="utf-8")


def _config(data_root: Path, run_dir: Path) -> ProjectConfig:
    config = ProjectConfig(
        model=PrometheusModelConfig(
            encoder_dims=[8, 16, 32, 64],
            encoder_depths=[1, 1, 1, 1],
            tissue_decoder_depths=[1, 1, 1],
        ),
        data=DataConfig(
            root=str(data_root),
            image_size=IMAGE_SIZE,
            validation_fraction=0.34,
            split_manifest=str(run_dir / "split.json"),
        ),
        trainer=EngineConfig(
            epochs=EPOCHS,
            batch_size=2,
            num_workers=0,
            amp=False,
            warmup_epochs=1,
            log_interval=1,
            ema_decay=0.9,
        ),
        evaluation=EvaluationConfig(checkpoint_metric="tissue/micro_dice"),
        paths=PathsConfig(run_dir=str(run_dir)),
    )
    config.loss.class_weighting = True
    return config


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ProjectConfig, dict]:
    """Train for two epochs on a synthetic dataset and return the run directory."""
    base = tmp_path_factory.mktemp("run")
    data_root, run_dir = base / "puma", base / "run"
    _write_dataset(data_root)
    config = _config(data_root, run_dir)

    trainer = build_trainer(
        config,
        model=build_model(config),
        datamodule=build_datamodule(config),
        device=torch.device("cpu"),
    )
    metrics = trainer.fit()
    return run_dir, config, metrics


@pytest.mark.integration
@pytest.mark.parametrize("name", CHECKPOINTS)
def test_every_checkpoint_is_written(completed_run, name: str) -> None:
    run_dir, _, _ = completed_run
    assert (run_dir / name).is_file(), f"{name} missing: a finished run must leave a model"


@pytest.mark.integration
@pytest.mark.parametrize("name", SIDECARS)
def test_every_sidecar_is_written(completed_run, name: str) -> None:
    run_dir, _, _ = completed_run
    path = run_dir / name
    assert path.is_file(), f"{name} missing"
    assert path.stat().st_size > 0, f"{name} is empty"


@pytest.mark.integration
def test_the_checkpoint_loads_back_into_a_model(completed_run) -> None:
    run_dir, config, _ = completed_run
    checkpoint = load_engine_checkpoint(run_dir / "best_tissue.ckpt", torch.device("cpu"))
    model = build_model(config)

    model.load_state_dict(select_inference_state(checkpoint))

    assert checkpoint["epoch"] == EPOCHS - 1
    assert checkpoint["ema_state"] is not None, "EMA was enabled, so it must be stored"
    with torch.no_grad():
        output = model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert output.tissue_logits.shape == (1, config.model.num_tissue_classes, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.integration
def test_the_history_has_one_json_line_per_epoch(completed_run) -> None:
    run_dir, _, _ = completed_run
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]

    assert [record["epoch"] for record in records] == list(range(EPOCHS))
    assert all("tissue/micro_dice" in record for record in records)
    assert all("nuclei/macro_f1_summed" in record for record in records)


@pytest.mark.integration
def test_the_log_survives_the_process_and_is_reconstructable(completed_run) -> None:
    # This is the file the notebook loses on a Colab disconnect if it only ever printed.
    run_dir, _, _ = completed_run
    log = (run_dir / "train.log").read_text(encoding="utf-8")

    assert "run " in log, "the session header records which experiment ran"
    assert "device cpu" in log, "the session header records the device"
    assert "tissue weights:" in log, "the weights actually used must be recoverable"
    for epoch in range(EPOCHS):
        assert f"Epoch {epoch:03d}" in log
    assert "tissue_micro_dice=" in log
    assert log.count("necrosis=") >= EPOCHS, "the rare class must be visible every epoch"


@pytest.mark.integration
def test_per_class_tissue_dice_is_reported(completed_run) -> None:
    _, _, metrics = completed_run

    for name in ("tumor", "stroma", "epidermis", "necrosis", "blood_vessel"):
        assert f"tissue/micro_dice/{name}" in metrics
    assert 0.0 <= metrics["tissue/micro_dice"] <= 1.0


@pytest.mark.integration
def test_resuming_appends_to_the_same_log_and_does_not_retrain(completed_run) -> None:
    run_dir, config, _ = completed_run
    before = (run_dir / "train.log").read_text(encoding="utf-8")

    trainer = build_trainer(
        config,
        model=build_model(config),
        datamodule=build_datamodule(config),
        device=torch.device("cpu"),
    )
    trainer.fit(resume_from=run_dir / "last.ckpt")
    after = (run_dir / "train.log").read_text(encoding="utf-8")

    assert after.startswith(before), "a resume must append, never truncate the history"
    assert after.count("run ") == 2, "each session is marked in the same file"
    assert trainer.start_epoch == EPOCHS, "the schedule was already complete, so nothing reruns"
