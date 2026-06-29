"""Serializable training state independent from a specific trainer."""

from dataclasses import asdict, dataclass


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    best_tissue_metric: float = float("-inf")
    best_nuclei_metric: float = float("-inf")
    best_epoch: int = 0
    early_stopping_counter: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
