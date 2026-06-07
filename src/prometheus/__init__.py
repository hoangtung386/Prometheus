__version__ = "0.2.0"

from .losses import (
    BCEWithLogitsLoss,
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    MultiClassDiceLoss,
    TverskyLoss,
)
from .models.unet_dual import DualUNet
from .models.unet_tissue import UNet as UNetTissue

_TRAINING_NAMES = {"Trainer", "dice_score", "train_one_epoch", "validate", "warmup_cosine_lr"}
_VIS_NAMES = {"predict_sample", "show_prediction", "visualize_sample"}


def __getattr__(name):
    if name in _TRAINING_NAMES:
        from .training import (
            Trainer,
            dice_score,
            train_one_epoch,
            validate,
            warmup_cosine_lr,
        )
        return locals()[name]
    if name in _VIS_NAMES:
        from .visualization import predict_sample, show_prediction, visualize_sample
        return locals()[name]
    raise AttributeError(f"module 'prometheus' has no attribute {name!r}")


__all__ = [
    "DualUNet",
    "UNetTissue",
    "BCEWithLogitsLoss",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "MultiClassDiceLoss",
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
