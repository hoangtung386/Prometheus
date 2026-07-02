from __future__ import annotations

from prometheus.models.backbones.pretrained import _timm_key


def test_backbone_keys_map_to_timm_convnextv2_names() -> None:
    assert _timm_key("stem.0.weight") == "stem.0.weight"
    assert _timm_key("stem.1.bias") == "stem.1.bias"
    # Our downsample list sits between stages; timm attaches it to the *next* stage.
    assert _timm_key("downsamples.0.0.weight") == "stages.1.downsample.0.weight"
    assert _timm_key("downsamples.2.1.bias") == "stages.3.downsample.1.bias"
    # Block internals: our V2 names -> timm's conv_dw / mlp.fc* / mlp.grn.
    assert _timm_key("stages.0.0.dwconv.weight") == "stages.0.blocks.0.conv_dw.weight"
    assert _timm_key("stages.2.5.norm.bias") == "stages.2.blocks.5.norm.bias"
    assert _timm_key("stages.1.0.pwconv1.weight") == "stages.1.blocks.0.mlp.fc1.weight"
    assert _timm_key("stages.3.2.pwconv2.bias") == "stages.3.blocks.2.mlp.fc2.bias"
    assert _timm_key("stages.0.1.grn.gamma") == "stages.0.blocks.1.mlp.grn.weight"
    assert _timm_key("stages.0.1.grn.beta") == "stages.0.blocks.1.mlp.grn.bias"
