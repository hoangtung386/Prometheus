# Prometheus

Medical image segmentation with ConvNeXt U-Net + Transformer + Mixture of Experts.

## Architecture

Three model variants built on a **ConvNeXt-v2 U-Net** backbone:

| Model | Description |
|-------|-------------|
| **UNetTissue** (`unet_tissue.py`) | Standard ConvNeXt U-Net for tissue-only segmentation |
| **DualUNet** (`unet_dual.py`) | Dual-stream architecture: tissue stream + nuclei stream with cross-attention fusion via **EncoderTransformerStack** (6× blocks), each with Self-Attn → Dense FFN → Cross-Attn → Sparse MoE |

### Tissue Stream
- Standard ConvNeXt U-Net encoder (Stem + 4 stages with downsampling) + decoder (3 levels, symmetric skip connections)
- TissueAttentionEncoder encodes decoder features into bottleneck context for nuclei cross-attention
- Outputs tissue mask + high-res feature map

### Nuclei Stream
- ConvNeXt encoder → **EncoderTransformerStack** (6 blocks, each: Self-Attn → FFN → Cross-Attn → MoE) → decoder → nuclei mask
- Each block uses **Local-Global Attention**: 50/50 heads split between windowed (Swin-style) and full-sequence attention
- **Dense FFN** + **Sparse MoE** (16 experts, top-2 gating, Down/Up projections, load balancing loss)
- Tissue features pass through **stop-gradient** (`.detach()`) → TissueAttentionEncoder → cross-attention context, isolating nuclei gradients from the tissue stream

### Supporting Components

| Module | Source | Description |
|--------|--------|-------------|
| `ConvNeXtBlock` | `blocks/convnext_block.py` | Pre-norm ConvNeXt block: DWConv 7×7 → Permute → LN → Linear(×4) → GELU → GRN → Linear(÷4) → Permute → DropPath + residual |
| `DecoderBlock` | `blocks/decoder_block.py` | Upsample (ConvTranspose2d) → skip concat → SkipProj(1×1) → ConvNeXt-style DWConv + LN + FFN + GRN + residual |
| `LocalGlobalAttention` | `blocks/attention.py` | Multi-head attention split 50/50 into local (windowed, Swin-style) and global (full-sequence) heads. Supports self-attention and cross-attention |
| `Expert` | `blocks/moe.py` | Individual expert network: Linear(d_expert → d_ff) → SiLU → Linear(d_ff → d_expert) |
| `SparseMoE` | `blocks/moe.py` | Sparse mixture-of-experts: DownProj → top-k gating over N experts → aggregate → UpProj. Includes load balancing auxiliary loss |
| `EncoderTransformerBlock` | `blocks/transformer_block.py` | Single transformer block: Self-Attn → FFN → Cross-Attn → Sparse MoE, all with Pre-LN + residual + dropout |
| `EncoderTransformerStack` | `blocks/transformer_block.py` | Stack of N `EncoderTransformerBlock`s with shared context for cross-attention |
| `LayerNorm` | `utils/norm.py` | Custom LayerNorm supporting both `channels_last` (F.layer_norm) and `channels_first` (manual µ,σ) |
| `GRN` | `utils/norm.py` | Global Response Normalization: Gx = ‖x‖₂ over (H,W), Nx = Gx / mean(Gx), out = γ·(x·Nx) + β + x |

### Loss Functions

All loss functions are importable from the top-level package:

| Loss | Description |
|------|-------------|
| `BCEWithLogitsLoss` | Binary cross-entropy with optional pos_weight |
| `DiceLoss` | Soft Dice loss (sigmoid + Dice coefficient) |
| `FocalLoss` | Focal loss with tunable α, γ |
| `CombinedLoss` | Weighted BCE + Dice combination (binary, kept for backward compat) |
| `MultiClassDiceLoss` | Multi-class Dice via softmax + one-hot encoding |
| `MulticlassCombinedLoss` | **Default for PUMA**: CrossEntropy + MultiClassDice (softmax-based, multiclass mutually-exclusive) |
| `TverskyLoss` | Tversky loss with tunable α (FP penalty), β (FN penalty) |

## Installation

```bash
pip install -e .
```

## Usage

### Python API

```python
from prometheus import UNetTissue, DualUNet
from prometheus.config import ModelConfig
import torch

# Tissue segmentation only
cfg = ModelConfig(in_chans=3, num_classes=1)
model = UNetTissue(config=cfg)
x = torch.randn(2, 3, 256, 256)
out = model(x)  # (2, 1, 256, 256)

# Dual tissue + nuclei segmentation (multiclass)
model = DualUNet()  # default: 6 tissue + 11 nuclei classes
tissue_mask, nuclei_mask, _ = model(x)
# tissue_mask: (2, 6, 256, 256), nuclei_mask: (2, 11, 256, 256)
```

### Building custom models with blocks

```python
from prometheus.blocks import (
    ConvNeXtBlock, DecoderBlock, LocalGlobalAttention,
    EncoderTransformerBlock, SparseMoE,
)

# ConvNeXt feature extractor
block = ConvNeXtBlock(dim=128, drop_path=0.1)

# Local-Global attention
attn = LocalGlobalAttention(d_model=512, n_heads=8, window_size=16)

# Transformer block with MoE
xfmr = EncoderTransformerBlock(d_model=256, n_heads=8, d_ff=2048)
out, loss = xfmr(x, context=tissue_features)
```

## Development

```bash
# Lint
flake8 src/

# Test
python -m pytest tests/

# Train on PUMA dataset
python scripts/train_puma.py --data-root /path/to/puma --model-type DualUNet

# Smoke test with dummy data
python scripts/train_tissue.py
```

## Project Structure

```
src/prometheus/
├── __init__.py              # Package init, exports UNetTissue, DualUNet, all losses
├── config.py                # ModelConfig, TrainingConfig dataclasses
├── blocks/                  # Reusable building blocks
│   ├── __init__.py
│   ├── attention.py         # LocalGlobalAttention
│   ├── convnext_block.py    # ConvNeXtBlock
│   ├── decoder_block.py     # DecoderBlock
│   ├── moe.py               # Expert, SparseMoE
│   └── transformer_block.py # EncoderTransformerBlock, EncoderTransformerStack
├── models/                  # Complete models
│   ├── __init__.py
│   ├── _base_unet.py        # Shared build_encoder, build_decoder factories
│   ├── unet_tissue.py       # UNetTissue (wrapper: Encoder + Decoder)
│   └── unet_dual.py         # DualUNet (dual-stream tissue + nuclei)
├── losses/                  # Loss functions
│   ├── __init__.py
│   └── segmentation.py      # BCEWithLogitsLoss, DiceLoss, FocalLoss, CombinedLoss, MultiClassDiceLoss, MulticlassCombinedLoss, TverskyLoss
└── utils/                   # Utilities
    ├── __init__.py
    └── norm.py              # LayerNorm, GRN
```

## License

[MIT](LICENSE)
