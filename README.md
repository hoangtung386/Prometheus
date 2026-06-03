# Prometheus

Medical image segmentation with ConvNeXt U-Net.

## Architecture

The project implements two segmentation models built on a **ConvNeXt-based U-Net** backbone:

### UNetTissue
- Standard U-Net for tissue segmentation
- ConvNeXt-style encoder/decoder blocks (depthwise conv, LayerNorm, GELU, GRN)
- Hierarchical encoder with 4 stages + skip connections
- Symmetric decoder with transpose conv upsampling

### UNetNuclei
- Extended U-Net for nuclei segmentation with tissue mask guidance
- Dense encoder + MinkowskiEngine sparse encoder for tissue mask features
- Transformer-style bottleneck encoder
- Auxiliary modules: Local-Global Attention

### Supporting Components

| Module | Description |
|--------|-------------|
| `ConvNeXtBlock` | Pre-norm ConvNeXt block: DWConv → LN → Linear(×4) → GELU → GRN → Linear(÷4) → residual |
| `DecoderBlock` | Upsample + skip concat + ConvNeXtBlock |
| `LocalGlobalAttention` | Multi-head attention split into local (windowed) and global (full) heads |


## Installation

```bash
pip install -e .
```

For MinkowskiEngine support (required for UNetNuclei):
```bash
pip install -e .[minkowski]
```

## Usage

### Python API

```python
from prometheus import UNetTissue, UNetNuclei
from prometheus.config import ModelConfig
import torch

# Tissue segmentation
cfg = ModelConfig(in_chans=3, num_classes=1)
model = UNetTissue(config=cfg)
x = torch.randn(2, 3, 256, 256)
out = model(x)  # (2, 1, 256, 256)

# Nuclei segmentation (requires MinkowskiEngine)
cfg = ModelConfig(in_chans=3, num_classes=1)
model = UNetNuclei(config=cfg)
x = torch.randn(2, 3, 256, 256)
mask = torch.randn(2, 1024)  # tissue mask
out = model(x, mask)
```

### Building custom models with blocks

```python
from prometheus.blocks import ConvNeXtBlock, DecoderBlock, LocalGlobalAttention

# ConvNeXt feature extractor
block = ConvNeXtBlock(dim=128, drop_path=0.1)

# Local-Global attention
attn = LocalGlobalAttention(d_model=512, n_heads=8, window_size=16)
```

## Development

```bash
# Lint
flake8 src/

# Test
python -m pytest tests/

# Run with your own data
python scripts/train_tissue.py
```

## Project Structure

```
src/prometheus/
├── __init__.py              # Package init, exports UNetTissue, UNetNuclei
├── config.py                # ModelConfig dataclass
├── blocks/                  # Reusable building blocks
│   ├── convnext_block.py    # ConvNeXtBlock
│   ├── decoder_block.py     # DecoderBlock
│   ├── attention.py         # LocalGlobalAttention
│   └── minkowski_block.py   # MinkowskiConvNeXtBlock (requires ME)
├── models/                  # Complete models
│   ├── _base_unet.py        # Shared encoder/decoder factories
│   ├── unet_tissue.py       # UNetTissue
│   ├── unet_nuclei.py       # UNetNuclei (requires ME)
│   └── unet_dual.py         # DualUNet
└── utils/                   # Utilities
    ├── norm.py              # LayerNorm, GRN
    └── minkowski_utils.py   # MinkowskiGRN, MinkowskiDropPath, MinkowskiLayerNorm
```

## License

[MIT](LICENSE)
