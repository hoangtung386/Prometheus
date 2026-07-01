"""Frozen compatibility implementations loaded only when explicitly requested."""

from importlib import import_module

_DUAL_NAMES = {"DualUNet", "TissueAttentionEncoder", "TissueDecoder"}
_TISSUE_NAMES = {"Decoder", "Encoder", "UNetTissue"}


def __getattr__(name):
    if name in _DUAL_NAMES:
        return getattr(import_module("prometheus.legacy.dual_unet"), name)
    if name in _TISSUE_NAMES:
        module = import_module("prometheus.legacy.tissue_unet")
        return module.UNet if name == "UNetTissue" else getattr(module, name)
    raise AttributeError(f"module 'prometheus.legacy' has no attribute {name!r}")


__all__ = [*_DUAL_NAMES, *_TISSUE_NAMES]
