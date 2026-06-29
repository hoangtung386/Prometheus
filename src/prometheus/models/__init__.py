from ._base_unet import build_decoder, build_encoder, forward_decoder, forward_encoder
from .registry import create_model, register_model, registered_models
from .unet_dual import DualUNet, TissueAttentionEncoder, TissueDecoder
from .unet_tissue import Decoder, Encoder
from .unet_tissue import UNet as UNetTissue

register_model("tissue_convnext_unet", lambda config: UNetTissue(config))
register_model("legacy_dual_unet", lambda config: DualUNet(config))

__all__ = [
    "DualUNet",
    "UNetTissue",
    "Encoder",
    "Decoder",
    "TissueAttentionEncoder",
    "TissueDecoder",
    "build_encoder",
    "forward_encoder",
    "build_decoder",
    "forward_decoder",
    "create_model",
    "register_model",
    "registered_models",
]
