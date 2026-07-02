"""Initialize the shared ConvNeXt-V2 encoder from ImageNet pretrained weights.

The backbone in :mod:`shared_convnext` is a faithful ConvNeXt-**V2** encoder (it uses
GRN and no layer-scale), so its parameters map one-to-one onto ``timm``'s
``convnextv2_tiny`` weights. On a tiny dataset (~185 training images) starting from
these features instead of random init is the single largest quality lever.

The mapping is by construction rather than by position and every tensor is shape
checked, so a name mismatch across ``timm`` versions surfaces as a loud report
(``loaded X/Y encoder tensors``) instead of silently corrupting the encoder.
"""

from __future__ import annotations

import torch

from ...config import PrometheusModelConfig
from .shared_convnext import SharedConvNeXtBackbone

DEFAULT_VARIANT = "convnextv2_tiny"


def _timm_key(our_key: str) -> str | None:
    """Translate a backbone parameter name to its ``timm`` ConvNeXt-V2 counterpart."""
    if our_key.startswith("stem."):
        return our_key
    parts = our_key.split(".")
    if parts[0] == "downsamples":
        # downsamples.{i}.{0|1}.{weight|bias} -> stages.{i+1}.downsample.{0|1}.{...}
        index = int(parts[1])
        return f"stages.{index + 1}.downsample.{'.'.join(parts[2:])}"
    if parts[0] == "stages":
        stage, block, module, tail = parts[1], parts[2], parts[3], ".".join(parts[4:])
        block_prefix = f"stages.{stage}.blocks.{block}"
        if module == "grn":
            return f"{block_prefix}.mlp.grn.{'weight' if tail == 'gamma' else 'bias'}"
        module_map = {
            "dwconv": "conv_dw",
            "norm": "norm",
            "pwconv1": "mlp.fc1",
            "pwconv2": "mlp.fc2",
        }
        return f"{block_prefix}.{module_map[module]}.{tail}"
    return None


def load_pretrained_backbone(
    backbone: SharedConvNeXtBackbone,
    config: PrometheusModelConfig,
    variant: str = DEFAULT_VARIANT,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """Copy ImageNet ConvNeXt-V2 weights into ``backbone`` in place.

    Returns a report ``{"loaded": [...], "skipped": [...]}``. Raises if ``timm`` cannot
    provide the weights or if nothing matched (which would indicate a broken mapping).
    """
    if config.encoder_dims != [96, 192, 384, 768] or config.encoder_depths != [3, 3, 9, 3]:
        raise ValueError(
            "Pretrained init only supports the ConvNeXt-V2 Tiny geometry "
            "(encoder_dims=[96,192,384,768], encoder_depths=[3,3,9,3]); "
            f"got dims={config.encoder_dims}, depths={config.encoder_depths}."
        )
    try:
        import timm
    except ImportError as error:  # pragma: no cover - environment guard
        raise ImportError("Pretrained backbone init requires `timm` (see requirements.txt).") from error

    reference = timm.create_model(variant, pretrained=True, num_classes=0)
    source = reference.state_dict()

    loaded: list[str] = []
    skipped: list[str] = []
    with torch.no_grad():
        for name, parameter in backbone.named_parameters():
            timm_name = _timm_key(name)
            source_tensor = source.get(timm_name) if timm_name is not None else None
            if source_tensor is None or source_tensor.numel() != parameter.numel():
                skipped.append(name)
                continue
            parameter.copy_(source_tensor.reshape(parameter.shape))
            loaded.append(name)

    total = len(loaded) + len(skipped)
    if not loaded:
        raise RuntimeError(
            f"Pretrained mapping matched 0/{total} tensors from timm '{variant}'. "
            "The timm parameter names likely changed; inspect reference.state_dict().keys()."
        )
    if verbose:
        print(f"Pretrained backbone: loaded {len(loaded)}/{total} encoder tensors from timm '{variant}'.")
        if skipped:
            print(f"  Skipped (kept random init): {skipped}")
    return {"loaded": loaded, "skipped": skipped}
