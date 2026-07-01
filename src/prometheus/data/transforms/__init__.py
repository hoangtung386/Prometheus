"""Task-aware image and target transformations."""

from .common import (
    Compose,
    Normalize,
    NormalizeTile,
    RandomBrightnessContrast,
    RandomChannelJitter,
    RandomGamma,
    RandomGaussianNoise,
)
from .multitask import (
    MultitaskCompose,
    NormalizeMultitask,
    RandomHorizontalFlipMultitask,
    RandomRotate90Multitask,
    RandomVerticalFlipMultitask,
    TransformSample,
    multitask_train_transform,
    multitask_validation_transform,
)
from .segmentation import (
    ElasticDeformation,
    RandomHorizontalFlip,
    RandomRotate90,
    RandomVerticalFlip,
    test_transform,
    train_transform,
    val_transform,
)


def collate_puma(batch):
    """Compatibility alias for the legacy segmentation collator."""

    from ..puma.loaders import collate_segmentation

    return collate_segmentation(batch)


__all__ = [
    "Compose",
    "ElasticDeformation",
    "Normalize",
    "NormalizeTile",
    "RandomBrightnessContrast",
    "RandomChannelJitter",
    "RandomGamma",
    "RandomGaussianNoise",
    "RandomHorizontalFlip",
    "RandomRotate90",
    "RandomVerticalFlip",
    "collate_puma",
    "test_transform",
    "train_transform",
    "val_transform",
    "MultitaskCompose",
    "NormalizeMultitask",
    "RandomHorizontalFlipMultitask",
    "RandomRotate90Multitask",
    "RandomVerticalFlipMultitask",
    "TransformSample",
    "multitask_train_transform",
    "multitask_validation_transform",
]
