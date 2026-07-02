"""Small stable composition API shared by CLI and Colab."""

from __future__ import annotations

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


def load_config(path) -> ProjectConfig:
    return load_project_config(path)


def build_datamodule(config: ProjectConfig):
    return create_multitask_dataloaders(
        root=config.data.root,
        image_size=config.data.image_size,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        validation_fraction=config.data.validation_fraction,
        seed=config.data.split_seed,
        split_manifest_path=config.data.split_manifest,
        pin_memory=True,
        strict_labels=config.data.strict_labels,
    )


def build_kfold_datamodule(
    config: ProjectConfig,
    fold_index: int,
    num_folds: int = 5,
    kfold_manifest_path=None,
):
    """Train/validation loaders for one fold of a k-fold split over the whole dataset."""
    return create_multitask_kfold_dataloaders(
        root=config.data.root,
        image_size=config.data.image_size,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        num_folds=num_folds,
        fold_index=fold_index,
        seed=config.data.split_seed,
        kfold_manifest_path=kfold_manifest_path,
        pin_memory=True,
        strict_labels=config.data.strict_labels,
    )


def build_model(config: ProjectConfig, pretrained: bool = False) -> PrometheusNet:
    """Build PrometheusNet, optionally seeding the encoder with ImageNet weights.

    ``pretrained`` is off by default so CLI/tests stay offline; the training notebook
    turns it on. It is a build-time choice, not part of the architecture identity, so
    it does not affect checkpoint compatibility.
    """
    model = PrometheusNet(config.model)
    if pretrained:
        load_pretrained_backbone(model.backbone, config.model)
    return model


def build_criterion(config: ProjectConfig) -> PrometheusMultitaskLoss:
    """Build the configured training loss without leaking config-only fields."""
    weight_fields = LossWeights.__dataclass_fields__
    weights = LossWeights(**{name: getattr(config.loss, name) for name in weight_fields})
    return PrometheusMultitaskLoss(
        config.model.num_nucleus_types,
        config.model.nuclei_feature_stride,
        weights,
        gaussian_radius=config.loss.gaussian_radius,
    )


def build_trainer(config: ProjectConfig, model=None, datamodule=None, device=None) -> PrometheusTrainer:
    model = model or build_model(config)
    train_loader, validation_loader = datamodule or build_datamodule(config)
    return PrometheusTrainer(model, train_loader, validation_loader, config, device)


def load_predictor(config: ProjectConfig, checkpoint_path, device=None) -> PrometheusPredictor:
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
