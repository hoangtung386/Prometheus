"""Stable composition root shared by the CLI, the notebook and the submission runtime.

Every entry point builds its objects through this module so there is exactly one place
where a config becomes a model, a datamodule and a trainer. Anything importing internals
directly is a bug waiting for the internals to move.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import ProjectConfig, load_project_config
from .data import create_multitask_dataloaders, create_multitask_kfold_dataloaders
from .engine import (
    PrometheusTrainer,
    assert_checkpoint_compatible,
    load_engine_checkpoint,
    select_inference_state,
)
from .inference import PrometheusPredictor
from .losses import LossWeights, PrometheusMultitaskLoss
from .models import PrometheusNet
from .models.backbones import load_pretrained_backbone

__all__ = [
    "build_criterion",
    "build_datamodule",
    "build_kfold_datamodule",
    "build_model",
    "build_trainer",
    "load_config",
    "load_predictor",
]


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate an experiment TOML file."""
    return load_project_config(path)


def build_datamodule(config: ProjectConfig):
    """Return ``(train_loader, validation_loader)`` for the single holdout split."""
    return create_multitask_dataloaders(
        root=config.data.root,
        image_size=config.data.image_size,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        validation_fraction=config.data.validation_fraction,
        seed=config.data.split_seed,
        split_manifest_path=config.data.split_manifest,
        strict_labels=config.data.strict_labels,
    )


def build_kfold_datamodule(
    config: ProjectConfig,
    fold_index: int,
    num_folds: int = 5,
    kfold_manifest_path: str | Path | None = None,
):
    """Return ``(train_loader, validation_loader)`` for one fold of a k-fold split."""
    return create_multitask_kfold_dataloaders(
        root=config.data.root,
        image_size=config.data.image_size,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        num_folds=num_folds,
        fold_index=fold_index,
        seed=config.data.split_seed,
        kfold_manifest_path=kfold_manifest_path,
        strict_labels=config.data.strict_labels,
    )


def build_model(config: ProjectConfig, pretrained: bool = False) -> PrometheusNet:
    """Build :class:`~prometheus.models.PrometheusNet`, optionally seeding the encoder.

    ``pretrained`` is off by default so the CLI and the test suite stay offline; the
    training notebook turns it on. It is a build-time choice rather than part of the
    architecture identity, so it does not affect checkpoint compatibility.
    """
    model = PrometheusNet(config.model)
    if pretrained:
        load_pretrained_backbone(model.backbone, config.model, variant=config.model.pretrained_variant)
    return model


def build_criterion(config: ProjectConfig) -> PrometheusMultitaskLoss:
    """Build the configured loss without class weights.

    Class weights require a pass over the training data, so they are resolved and cached by
    :class:`~prometheus.engine.PrometheusTrainer`. Use this for evaluation, where the
    reported loss should not depend on a fold's class histogram.
    """
    return PrometheusMultitaskLoss(
        config.model.num_nucleus_types,
        config.model.nuclei_feature_stride,
        LossWeights(**{name: getattr(config.loss, name) for name in LossWeights.field_names()}),
        gaussian_radius=config.loss.gaussian_radius,
    )


def build_trainer(
    config: ProjectConfig,
    model: PrometheusNet | None = None,
    datamodule: tuple | None = None,
    device: torch.device | None = None,
) -> PrometheusTrainer:
    """Build a trainer, constructing the model and datamodule when not supplied."""
    model = model or build_model(config)
    train_loader, validation_loader = datamodule or build_datamodule(config)
    return PrometheusTrainer(model, train_loader, validation_loader, config, device)


def load_predictor(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> PrometheusPredictor:
    """Load a checkpoint into a predictor, preferring its EMA weights when present."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    checkpoint = load_engine_checkpoint(checkpoint_path, device)
    assert_checkpoint_compatible(checkpoint, config)
    model.load_state_dict(select_inference_state(checkpoint))
    return PrometheusPredictor(
        model,
        device,
        config.model.nuclei_feature_stride,
        config.postprocess.confidence_threshold,
        config.postprocess.max_detections,
        config.postprocess.local_max_kernel,
    )
