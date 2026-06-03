from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    in_chans: int = 3
    num_classes: int = 1
    encoder_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
    drop_path_rate: float = 0.1
    D: int = 2


DEFAULT_CONFIG = ModelConfig()
