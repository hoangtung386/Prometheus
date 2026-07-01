"""Configuration contracts retained only for legacy semantic experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.common import DataConfig, EvaluationConfig


@dataclass
class ModelConfig:
    in_chans: int = 3
    num_classes: int | None = None
    num_tissue_classes: int = 6
    num_nuclei_classes: int = 11
    encoder_dims: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_depths: list[int] = field(default_factory=lambda: [3, 3, 9, 3])
    drop_path_rate: float = 0.1
    n_heads: int = 8
    d_ff: int = 3072
    d_expert: int = 256
    window_size: int = 8
    num_transformer_blocks: int = 2
    num_experts: int = 16
    moe_top_k: int = 2
    use_tissue_context: bool = True
    use_moe: bool = False

    def validate(self) -> None:
        if self.in_chans <= 0:
            raise ValueError("in_chans must be positive")
        if len(self.encoder_dims) != 4 or len(self.encoder_depths) != 4:
            raise ValueError("The legacy U-Net implementation requires four encoder stages")
        if any(value <= 0 for value in self.encoder_dims + self.encoder_depths):
            raise ValueError("Encoder dimensions and depths must be positive")
        if self.moe_top_k <= 0 or self.moe_top_k > self.num_experts:
            raise ValueError("moe_top_k must be in [1, num_experts]")
        if self.n_heads <= 0 or self.encoder_dims[-1] % self.n_heads != 0:
            raise ValueError("Bottleneck dimension must be divisible by n_heads")
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError("num_classes must be positive when provided")
        if self.num_tissue_classes <= 0 or self.num_nuclei_classes <= 0:
            raise ValueError("Task class counts must be positive")


@dataclass
class TrainingConfig:
    model_type: str = "DualUNet"
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_epochs: int = 5
    gradient_clip_norm: float | None = 1.0
    scheduler_min_lr: float = 1e-6
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 10
    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    moe_loss_weight: float = 0.01
    use_class_weights: bool = True
    class_weight_power: float = 0.5
    early_stopping_patience: int | None = None
    early_stopping_monitor: str = "combined"
    tissue_context_warmup_epochs: int = 0
    image_size: int = 1024
    num_tissue_classes: int = 6
    num_nuclei_classes: int = 11
    data_root: str = ""
    test_split: float = 0.1

    def validate(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.lr <= 0 or self.scheduler_min_lr < 0:
            raise ValueError("Learning rates must be non-negative and lr must be positive")
        if self.early_stopping_monitor not in {"combined", "tissue", "nuclei"}:
            raise ValueError("Unsupported early_stopping_monitor")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < self.test_split < 1.0:
            raise ValueError("test_split must be between zero and one")
        if self.num_tissue_classes < 1 or self.num_nuclei_classes < 1:
            raise ValueError("num_tissue_classes and num_nuclei_classes must be positive")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.evaluation.validate()
