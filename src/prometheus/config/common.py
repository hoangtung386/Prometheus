"""Configuration shared by production and compatibility workflows."""

from dataclasses import dataclass


@dataclass
class DataConfig:
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
class EvaluationConfig:
    nuclei_radius_px: float = 15.0
    checkpoint_metric: str = "combined"
    track: str = "track2"

    def validate(self) -> None:
        if self.nuclei_radius_px <= 0:
            raise ValueError("nuclei_radius_px must be positive")
        if self.track not in {"track1", "track2"}:
            raise ValueError("track must be track1 or track2")
