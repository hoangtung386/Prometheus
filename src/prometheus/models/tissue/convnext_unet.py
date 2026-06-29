"""Canonical import path for the ConvNeXt U-Net tissue model."""

from ..unet_tissue import UNet as ConvNeXtUNet

UNetTissue = ConvNeXtUNet

__all__ = ["ConvNeXtUNet", "UNetTissue"]
