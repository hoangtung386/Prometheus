"""Configuration contracts. Every field is validated; unknown TOML keys are rejected.

A config is part of the checkpoint: :func:`prometheus.engine.assert_checkpoint_compatible`
compares ``ProjectConfig.model`` against the config stored in a checkpoint, so any change
to :class:`PrometheusModelConfig` invalidates existing weights by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "CHECKPOINT_METRICS",
    "DataConfig",
    "EngineConfig",
    "EvaluationConfig",
    "LossConfig",
    "OptimizerConfig",
    "PathsConfig",
    "PostprocessConfig",
    "ProjectConfig",
    "PrometheusModelConfig",
]

CHECKPOINT_METRICS = frozenset({"nuclei/macro_f1_summed", "tissue/micro_dice"})
"""Metrics that may drive ``best_primary.ckpt`` selection.

``tissue/micro_dice`` is the official leaderboard tissue metric; select on it when the run
targets tissue. ``best_tissue.ckpt`` always tracks it regardless of this setting.
"""

_MODEL_NAME = "prometheus_multitask_v1"
_ENCODER_STAGES = 4
_DECODER_LEVELS = 3


@dataclass
class PrometheusModelConfig:
    """Architecture identity. Changing any field invalidates existing checkpoints."""

    name: str = _MODEL_NAME
    in_channels: int = 3
    num_tissue_classes: int = 6
    num_nucleus_types: int = 10
    encoder_dims: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_depths: list[int] = field(default_factory=lambda: [3, 3, 9, 3])
    tissue_decoder_depths: list[int] = field(default_factory=lambda: [1, 2, 2])
    drop_path_rate: float = 0.1
    context_enabled: bool = True
    nuclei_feature_stride: int = 4
    pretrained_variant: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"

    def validate(self) -> None:
        if self.name != _MODEL_NAME:
            raise ValueError(f"Unsupported model name: {self.name}")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if len(self.encoder_dims) != _ENCODER_STAGES or len(self.encoder_depths) != _ENCODER_STAGES:
            raise ValueError(f"encoder_dims and encoder_depths must contain {_ENCODER_STAGES} stages")
        if len(self.tissue_decoder_depths) != _DECODER_LEVELS:
            raise ValueError(f"tissue_decoder_depths must contain {_DECODER_LEVELS} levels")
        if any(value <= 0 for value in (*self.encoder_dims, *self.encoder_depths, *self.tissue_decoder_depths)):
            raise ValueError("Model dimensions and depths must be positive")
        if self.num_tissue_classes <= 0 or self.num_nucleus_types <= 0:
            raise ValueError("Task class counts must be positive")
        if self.nuclei_feature_stride not in {4, 8}:
            raise ValueError("nuclei_feature_stride must be 4 or 8")
        if not self.pretrained_variant.strip():
            raise ValueError("pretrained_variant cannot be empty")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")


@dataclass
class DataConfig:
    """Dataset location and the model input geometry."""

    root: str = ""
    image_size: int = 1024
    validation_fraction: float = 0.1
    split_seed: int = 42
    strict_labels: bool = True
    split_manifest: str = "runs/splits/puma.json"

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")


@dataclass
class OptimizerConfig:
    """AdamW hyper-parameters, including the reduced encoder learning rate."""

    lr: float = 2e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    backbone_lr_multiplier: float = 0.1
    """Learning-rate factor for the pretrained encoder relative to the fresh heads."""

    def validate(self) -> None:
        if self.lr <= 0 or self.weight_decay < 0 or self.eps <= 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 < self.backbone_lr_multiplier <= 1:
            raise ValueError("backbone_lr_multiplier must be in (0, 1]")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("Optimizer betas must contain two values in [0, 1)")


@dataclass
class EngineConfig:
    """Training loop sizing: schedule length, batching, precision and averaging."""

    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    amp: bool = True
    gradient_accumulation: int = 1
    gradient_clip_norm: float | None = 1.0
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    log_interval: int = 10
    ema_decay: float = 0.0
    """Exponential moving average decay for the evaluated weights; ``0`` disables it."""

    def validate(self, lr: float) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("Invalid trainer size configuration")
        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError("warmup_epochs must be between zero and epochs")
        if self.gradient_accumulation <= 0 or self.log_interval <= 0:
            raise ValueError("gradient_accumulation and log_interval must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if not 0 <= self.min_lr <= lr:
            raise ValueError("min_lr must be between zero and the optimizer learning rate")


@dataclass
class LossConfig:
    """Per-term loss weights and the class-weighting policy."""

    tissue_ce: float = 1.0
    tissue_dice: float = 1.0
    center_focal: float = 1.0
    nuclei_class: float = 1.0
    offset: float = 1.0
    size: float = 0.1
    gaussian_radius: int = 2
    class_weighting: bool = False
    class_weight_power: float = 0.5
    """Exponent on the inverse class frequency; ``0.5`` is square-root inverse frequency.

    ``1.0`` reproduces plain inverse frequency, which on this dataset spans 420:1 and
    destabilises early training. See :mod:`prometheus.losses.class_weights`.
    """

    _NON_WEIGHT_FIELDS = ("gaussian_radius", "class_weighting", "class_weight_power")

    def validate(self) -> None:
        weights = [value for name, value in self.__dict__.items() if name not in self._NON_WEIGHT_FIELDS]
        if any(value < 0 for value in weights):
            raise ValueError("Loss weights must be non-negative")
        if self.gaussian_radius < 0:
            raise ValueError("gaussian_radius must be non-negative")
        if not 0.0 < self.class_weight_power <= 1.0:
            raise ValueError("class_weight_power must be in (0, 1]")


@dataclass
class EvaluationConfig:
    """Validation metric settings and the target challenge track."""

    nuclei_radius_px: float = 15.0
    """Centroid hit radius of the official evaluator, in source-image pixels."""

    checkpoint_metric: str = "nuclei/macro_f1_summed"
    track: str = "track2"

    def validate(self) -> None:
        if self.nuclei_radius_px <= 0:
            raise ValueError("nuclei_radius_px must be positive")
        if self.track not in {"track1", "track2"}:
            raise ValueError("track must be track1 or track2")
        if self.checkpoint_metric not in CHECKPOINT_METRICS:
            raise ValueError(f"checkpoint_metric must be one of {sorted(CHECKPOINT_METRICS)}")


@dataclass
class PathsConfig:
    """Filesystem destinations for a run."""

    run_dir: str = "runs/default"


@dataclass
class PostprocessConfig:
    """Nuclei decoding thresholds applied after the forward pass."""

    confidence_threshold: float = 0.25
    max_detections: int = 1000
    local_max_kernel: int = 3

    def validate(self) -> None:
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between zero and one")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if self.local_max_kernel <= 0 or self.local_max_kernel % 2 == 0:
            raise ValueError("local_max_kernel must be a positive odd integer")


@dataclass
class ProjectConfig:
    """Fully resolved configuration of one experiment."""

    model: PrometheusModelConfig = field(default_factory=PrometheusModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: EngineConfig = field(default_factory=EngineConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    name: str = "prometheus_experiment"
    seed: int = 42

    def validate(self) -> None:
        """Validate every section. Raises ``ValueError`` on the first problem found.

        Called by the loader and again by the trainer, so a config built in Python (as the
        notebook does) is held to the same rules as one parsed from TOML.
        """
        self.model.validate()
        self.data.validate()
        self.optimizer.validate()
        self.trainer.validate(self.optimizer.lr)
        self.loss.validate()
        self.evaluation.validate()
        self.postprocess.validate()
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty")
