"""Instance-aware multitask transformations."""
from .multitask import (
    MultitaskCompose,
    NormalizeMultitask,
    RandomBrightnessContrastMultitask,
    RandomGammaMultitask,
    RandomGaussianNoiseMultitask,
    RandomHorizontalFlipMultitask,
    RandomRotate90Multitask,
    RandomStainJitterMultitask,
    RandomVerticalFlipMultitask,
    TransformSample,
    multitask_train_transform,
    multitask_validation_transform,
)

__all__ = [
    "MultitaskCompose",
    "NormalizeMultitask",
    "RandomBrightnessContrastMultitask",
    "RandomGammaMultitask",
    "RandomGaussianNoiseMultitask",
    "RandomHorizontalFlipMultitask",
    "RandomRotate90Multitask",
    "RandomStainJitterMultitask",
    "RandomVerticalFlipMultitask",
    "TransformSample",
    "multitask_train_transform",
    "multitask_validation_transform",
]
