"""Production configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import DataConfig, EvaluationConfig


@dataclass
class PrometheusModelConfig:
    name: str = "prometheus_multitask_v1"
    in_channels: int = 3
    num_tissue_classes: int = 6
    num_nucleus_types: int = 10
    encoder_dims: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_depths: list[int] = field(default_factory=lambda: [3, 3, 9, 3])
    tissue_decoder_depths: list[int] = field(default_factory=lambda: [1, 2, 2])
    drop_path_rate: float = 0.1
    context_enabled: bool = True
    nuclei_feature_stride: int = 4

    def validate(self) -> None:
        if self.name != "prometheus_multitask_v1":
            raise ValueError(f"Unsupported model name: {self.name}")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if len(self.encoder_dims) != 4 or len(self.encoder_depths) != 4:
            raise ValueError("encoder_dims and encoder_depths must contain four stages")
        if len(self.tissue_decoder_depths) != 3:
            raise ValueError("tissue_decoder_depths must contain three levels")
        values = self.encoder_dims + self.encoder_depths + self.tissue_decoder_depths
        if any(value <= 0 for value in values):
            raise ValueError("Model dimensions and depths must be positive")
        if self.num_tissue_classes <= 0 or self.num_nucleus_types <= 0:
            raise ValueError("Task class counts must be positive")
        if self.nuclei_feature_stride not in {4, 8}:
            raise ValueError("nuclei_feature_stride must be 4 or 8")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")


@dataclass
class OptimizerConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class EngineConfig:
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    amp: bool = True
    gradient_accumulation: int = 1
    gradient_clip_norm: float | None = 1.0
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    log_interval: int = 10
    ema_decay: float = 0.0  # exponential moving average of weights; 0 disables it


@dataclass
class LossConfig:
    tissue_ce: float = 1.0
    tissue_dice: float = 1.0
    center_focal: float = 1.0
    nuclei_class: float = 1.0
    offset: float = 1.0
    size: float = 0.1
    gaussian_radius: int = 2
    class_weighting: bool = False  # inverse-frequency class weights for tissue + nuclei


@dataclass
class PathsConfig:
    run_dir: str = "runs/default"


@dataclass
class PostprocessConfig:
    confidence_threshold: float = 0.25
    max_detections: int = 1000
    local_max_kernel: int = 3


@dataclass
class ProjectConfig:
    model: PrometheusModelConfig = field(default_factory=PrometheusModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: EngineConfig = field(default_factory=EngineConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(
        default_factory=lambda: EvaluationConfig(checkpoint_metric="nuclei/macro_f1_summed")
    )
    paths: PathsConfig = field(default_factory=PathsConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    name: str = "prometheus_experiment"
    seed: int = 42

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.evaluation.validate()
        if self.optimizer.lr <= 0 or self.optimizer.weight_decay < 0 or self.optimizer.eps <= 0:
            raise ValueError("Invalid optimizer configuration")
        if len(self.optimizer.betas) != 2 or not all(0 <= beta < 1 for beta in self.optimizer.betas):
            raise ValueError("Optimizer betas must contain two values in [0, 1)")
        if self.trainer.epochs <= 0 or self.trainer.batch_size <= 0 or self.trainer.num_workers < 0:
            raise ValueError("Invalid trainer size configuration")
        if not 0 <= self.trainer.warmup_epochs <= self.trainer.epochs:
            raise ValueError("warmup_epochs must be between zero and epochs")
        if self.trainer.gradient_accumulation <= 0 or self.trainer.log_interval <= 0:
            raise ValueError("gradient_accumulation and log_interval must be positive")
        if not 0.0 <= self.trainer.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.trainer.gradient_clip_norm is not None and self.trainer.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if not 0 <= self.trainer.min_lr <= self.optimizer.lr:
            raise ValueError("min_lr must be between zero and the optimizer learning rate")
        non_weight_fields = {"gaussian_radius", "class_weighting"}
        loss_weights = {
            name: value
            for name, value in self.loss.__dict__.items()
            if name not in non_weight_fields
        }
        if any(value < 0 for value in loss_weights.values()) or self.loss.gaussian_radius < 0:
            raise ValueError("Loss weights and gaussian_radius must be non-negative")
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty")
        supported_metrics = {"nuclei/macro_f1_summed", "tissue/dice_mean_fg"}
        if self.evaluation.checkpoint_metric not in supported_metrics:
            raise ValueError(f"checkpoint_metric must be one of {sorted(supported_metrics)}")
        if not 0 < self.postprocess.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between zero and one")
        if self.postprocess.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        kernel = self.postprocess.local_max_kernel
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("local_max_kernel must be a positive odd integer")
