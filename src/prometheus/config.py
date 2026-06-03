from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    in_chans: int = 3
    num_classes: int = 1
    encoder_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
    drop_path_rate: float = 0.1
    D: int = 2

    n_heads: int = 8
    d_ff: int = 3072
    d_expert: int = 256
    window_size: int = 4
    num_transformer_blocks: int = 6
    num_experts: int = 512
    moe_top_k: int = 8
    moe_capacity_factor: float = 1.25


DEFAULT_CONFIG = ModelConfig()


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_epochs: int = 5
    gradient_clip_norm: Optional[float] = 1.0
    scheduler: str = "cosine"
    scheduler_min_lr: float = 1e-6
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 10
    val_interval: int = 1


DEFAULT_TRAINING_CONFIG = TrainingConfig()
