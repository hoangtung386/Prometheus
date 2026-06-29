__version__ = "0.3.0"

from importlib import import_module

from .losses import (
    BCEWithLogitsLoss,
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    MulticlassCombinedLoss,
    MultiClassDiceLoss,
    TverskyLoss,
)
from .metrics import SegmentationEvaluator
from .models.multitask import DualUNet
from .models.tissue import UNetTissue

_TRAINING_NAMES = {
    "Trainer",
    "dice_score",
    "train_one_epoch",
    "validate",
    "warmup_cosine_lr",
}
_VIS_NAMES = {"predict_sample", "show_prediction", "visualize_sample"}


def __getattr__(name):
    if name in _TRAINING_NAMES:
        return getattr(import_module(".training", __name__), name)
    if name in _VIS_NAMES:
        return getattr(import_module(".visualization", __name__), name)
    raise AttributeError(f"module 'prometheus' has no attribute {name!r}")


__all__ = [
    "DualUNet",
    "UNetTissue",
    "SegmentationEvaluator",
    "BCEWithLogitsLoss",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "MultiClassDiceLoss",
    "MulticlassCombinedLoss",
    "TverskyLoss",
    "Trainer",
    "dice_score",
    "train_one_epoch",
    "validate",
    "warmup_cosine_lr",
    "predict_sample",
    "show_prediction",
    "visualize_sample",
]
