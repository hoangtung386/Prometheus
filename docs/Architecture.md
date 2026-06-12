# Prometheus Architecture

## Overview

Prometheus is a medical image segmentation framework built on a **ConvNeXt-v2 U-Net** backbone. It provides two model variants:

| Model | Description |
|-------|-------------|
| **UNetTissue** | Standard ConvNeXt U-Net for tissue segmentation |
| **DualUNet** | Dual-stream architecture: tissue stream + nuclei stream with cross-attention fusion |

## Core Configuration

```python
ModelConfig:
  in_chans: 3
  num_classes: 1              # used only by UNetTissue output head
  num_tissue_classes: 6       # used by DualUNet tissue head
  num_nuclei_classes: 11      # used by DualUNet nuclei head
  encoder_dims: [96, 192, 384, 768]
  encoder_depths: [3, 3, 9, 3]
  drop_path_rate: 0.1
  D: 2
  window_size: 8              # local window size for attention
  num_transformer_blocks: 6
  num_experts: 16
  moe_top_k: 2
```

## Model Hierarchy

```mermaid
classDiagram
    class ConvNeXtBlock {
        +dwconv: Conv2d(dim, dim, 7x7, groups=dim)
        +norm: LayerNorm
        +pwconv1: Linear(dim, 4*dim)
        +act: GELU
        +grn: GRN(4*dim)
        +pwconv2: Linear(4*dim, dim)
        +drop_path: DropPath
        +forward(x) Tensor
    }

    class DecoderBlock {
        +upsample: ConvTranspose2d (optional)
        +skip_proj: Conv2d(2*dim, dim, 1x1)
        +dwconv: Conv2d(dim, dim, 7x7, groups=dim)
        +norm: LayerNorm
        +pwconv1: Linear(dim, 4*dim)
        +act: GELU
        +grn: GRN(4*dim)
        +pwconv2: Linear(4*dim, dim)
        +drop_path: DropPath
        +forward(x, skip) Tensor
    }

    class LocalGlobalAttention {
        +n_heads: int
        +local_heads: int (n_heads//2)
        +global_heads: int
        +window_size: int
        +d_model: int
        +d_head: int
        +q_proj, k_proj, v_proj: Linear
        +out_proj: Linear
        +forward(x, context?) Tensor
    }

    class EncoderTransformerBlock {
        +self_attn: LocalGlobalAttention
        +ffn: Sequential(Linear, GELU, Linear)
        +cross_attn: LocalGlobalAttention
        +moe: SparseMoE
        +norm1..8: LayerNorm
        +dropout: Dropout
        +forward(x, context?) Tuple[Tensor, Tensor]
    }

    class EncoderTransformerStack {
        +blocks: ModuleList[EncoderTransformerBlock]
        +forward(x, context?) Tuple[Tensor, Tensor]
    }

    class Expert {
        +fc1: Linear(d_expert, d_ff)
        +act: SiLU
        +fc2: Linear(d_ff, d_expert)
        +forward(x) Tensor
    }

    class SparseMoE {
        +down_proj: Linear(d_model, d_expert)
        +up_proj: Linear(d_expert, d_model)
        +gate: Linear(d_model, num_experts)
        +experts: ModuleList[Expert]
        +forward(x) Tuple[Tensor, Tensor]
    }

```

## UNetTissue

**File:** `src/prometheus/models/unet_tissue.py`

A straightforward ConvNeXt U-Net with symmetric encoder-decoder and skip connections.

```mermaid
flowchart TB
    subgraph Input["Input (B, 3, H, W)"]
        I[" "]
    end

    subgraph Encoder["ConvNeXt Encoder"]
        S["Stem<br/>Conv2d(4x4, s4) + LayerNorm<br/>(B, 96, H/4, W/4)"]
        D1["Down 1<br/>LayerNorm + Conv2d(2x2, s2)<br/>(B, 192, H/8, W/8)"]
        D2["Down 2<br/>LayerNorm + Conv2d(2x2, s2)<br/>(B, 384, H/16, W/16)"]
        D3["Down 3<br/>LayerNorm + Conv2d(2x2, s2)<br/>(B, 768, H/32, W/32)"]
        S1["Stage 1<br/>3× ConvNeXtBlock<br/>(B, 96, H/4, W/4)"]
        S2["Stage 2<br/>3× ConvNeXtBlock<br/>(B, 192, H/8, W/8)"]
        S3["Stage 3<br/>9× ConvNeXtBlock<br/>(B, 384, H/16, W/16)"]
        S4["Stage 4<br/>3× ConvNeXtBlock<br/>(B, 768, H/32, W/32)"]
    end

    subgraph Decoder["ConvNeXt Decoder"]
        direction TB
        L0["Level 0<br/>(B, 384, H/16, W/16)<br/>Upsample + Skip + ConvNeXtBlock ×9"]
        L1["Level 1<br/>(B, 192, H/8, W/8)<br/>Upsample + Skip + ConvNeXtBlock ×3"]
        L2["Level 2<br/>(B, 96, H/4, W/4)<br/>Upsample + Skip + ConvNeXtBlock ×3"]
        OH["Output Head<br/>ConvTranspose2d(4x4, s4) + Conv2d(1x1)<br/>(B, num_classes, H, W)"]
    end

    I --> S
    S --> S1
    S1 --> D1
    D1 --> S2
    S2 --> D2
    D2 --> S3
    S3 --> D3
    D3 --> S4

    S4 -- "bottleneck (B, 768, H/32, W/32)" --> L0
    S3 -- "skip" --> L0
    L0 -- "skip" --> L1
    S2 --> L1
    L1 -- "skip" --> L2
    S1 --> L2
    L2 --> OH

    style Input fill:#e1f5fe
    style Encoder fill:#e8f5e9
    style Decoder fill:#fff3e0
```

### ConvNeXtBlock Detail

```mermaid
flowchart LR
    subgraph ConvNeXtBlock["ConvNeXtBlock(dim, drop_path)"]
        I["Input<br/>(B, C, H, W)"] --> DW["DWConv 7×7<br/>groups=C"]
        DW --> P1["Permute → (B, H, W, C)"]
        P1 --> LN["LayerNorm"]
        LN --> FC1["Linear(C → 4C)"]
        FC1 --> G["GELU"]
        G --> GRN["GRN(4C)"]
        GRN --> FC2["Linear(4C → C)"]
        FC2 --> P2["Permute → (B, C, H, W)"]
        P2 --> DP["DropPath"]
        DP --> ADD["+ Input"]
        ADD --> O["Output<br/>(B, C, H, W)"]
    end
```

### DecoderBlock Detail

```mermaid
flowchart LR
    subgraph DecoderBlock["DecoderBlock(dim, has_upsample)"]
        X["x (decoder)"] --> UPS["ConvTranspose2d<br/>(optional)"]
        SKIP["skip (encoder)"] --> CAT["Concat(dim=1)"]
        UPS --> CAT
        CAT --> SKP["Skip Proj<br/>Conv2d(2C → C, 1×1)"]
        SKP --> DW["DWConv 7×7<br/>groups=C"]
        DW --> P1["Permute → (B, H, W, C)"]
        P1 --> LN["LayerNorm"]
        LN --> FC1["Linear(C → 4C)"]
        FC1 --> G["GELU"]
        G --> GRN["GRN(4C)"]
        GRN --> FC2["Linear(4C → C)"]
        FC2 --> P2["Permute → (B, C, H, W)"]
        P2 --> DP["DropPath"]
        DP --> ADD["+ Input"]
        ADD --> O["Output"]
    end
```

## DualUNet

**File:** `src/prometheus/models/unet_dual.py`

A dual-stream architecture with **stop-gradient** isolation between tissue and nuclei streams. The tissue stream's feature map is encoded via `TissueAttentionEncoder` (2× stem + 3× 2× downsamples = 16× reduction) and fused into the nuclei bottleneck through an **EncoderTransformerStack** (6× blocks, each with Self-Attn → FFN → Cross-Attn → Sparse MoE with 16 experts, top-2).

```mermaid
flowchart TB
    subgraph Input["Input (B, 3, H, W)"]
        I[" "]
    end

    subgraph Tissue["Tissue Stream"]
        direction TB
        TE["Encoder<br/>Stem + 4 ConvNeXt Stages"]
        TD["TissueDecoder<br/>3 Decoder Levels"]
        OH1["Mask Head<br/>ConvTranspose2d(4x4) + Conv2d(1x1)"]
        FH["Feature Head<br/>ConvTranspose2d(4x4)<br/>(B, 96, H, W)"]
    end

    subgraph Bridge["Tissue → Nuclei Bridge"]
        SG["stop_gradient()<br/>detach()"]
        TAE["TissueAttentionEncoder<br/>Stem(s2) + 3 Downsamples(s2)<br/>→ (B, 768, H/16, W/16)"]
    end

    subgraph Nuclei["Nuclei Stream"]
        direction TB
        NE["Encoder<br/>Stem + 4 ConvNeXt Stages"]
        TX["EncoderTransformerStack<br/>6× blocks<br/>Self-Attn → FFN → Cross-Attn → MoE"]
        ND["Decoder<br/>3 Decoder Levels"]
        OH2["Output Head<br/>ConvTranspose2d(4x4) + Conv2d(1x1)"]
    end

    I --> TE
    TE -->|"skips"| TD
    TE -->|"bottleneck (B, 768, H/32, W/32)"| TD
    TD --> FH
    FH --> OH1
    OH1 --> TM["Tissue Mask"]

    FH --> SG
    SG --> TAE
    TAE -->|"context"| TX

    I --> NE
    NE -->|"bottleneck (B, 768, H/32, W/32)"| TX
    NE -->|"skips"| ND
    TX --> ND
    ND --> OH2
    OH2 --> NM["Nuclei Mask"]

    style Input fill:#e1f5fe
    style Tissue fill:#e8f5e9
    style Bridge fill:#fff8e1
    style Nuclei fill:#fce4ec
```

### EncoderTransformerBlock Detail

Each of the 6 transformer blocks in the stack follows this structure:

```mermaid
flowchart LR
    subgraph Block["EncoderTransformerBlock(d_model, n_heads, d_ff)"]
        IN["x: (B, L, D)"] --> SA["Self-Attention<br/>LocalGlobalAttention<br/>Q=K=V=x"]
        SA --> SA_RES["+ x<br/>(Pre-LN residual)"]
        SA_RES --> FFN["Dense FFN<br/>Linear(D → 4D) → GELU → Linear(4D → D)"]
        FFN --> FFN_RES["+ x<br/>(Pre-LN residual)"]
        FFN_RES --> CA["Cross-Attention<br/>LocalGlobalAttention<br/>Q=x, K=V=context"]
        CA --> CA_RES["+ x<br/>(Pre-LN residual)"]
        CA_RES --> MOE["Sparse MoE<br/>DownProj → top-k gate → experts → UpProj"]
        MOE --> MOE_RES["+ x<br/>(Pre-LN residual)"]
        MOE_RES --> OUT["output: (B, L, D)"]

        CTX["context: (B, S, D)<br/>(from TissueAttentionEncoder)"] -.-> CA
    end
```

## LocalGlobalAttention

**File:** `src/prometheus/blocks/attention.py`

Splits attention heads 50/50 into **local** (windowed, Swin-style) and **global** (full sequence). Supports both self-attention and cross-attention modes.

```mermaid
flowchart TB
    subgraph LGA["LocalGlobalAttention(d_model, n_heads, window_size)"]
        X["x: (B, L, D)"] --> Q["Q = x @ Wq"]
        CTX["context: (B, S, D)<br/>(optional)"] --> K["K = context @ Wk"]
        CTX --> V["V = context @ Wv"]

        X --> K2["K = x @ Wk (if no context)"]
        X --> V2["V = x @ Wv (if no context)"]

        Q --> SPLIT["Split Heads"]
        K --> SPLIT2["Split Heads"]
        K2 --> SPLIT2
        V --> SPLIT3["Split Heads"]
        V2 --> SPLIT3

        SPLIT --> L["Local Heads (n/2)"]
        SPLIT --> G["Global Heads (n/2)"]

        subgraph LocalAttn["Local Window Attention"]
            LW["Reshape →<br/>(B, n/2, num_windows, W, d_head)"]
            LW2["Window Attention<br/>(within each window)"]
            LW --> LW2
        end

        subgraph GlobalAttn["Global Full Attention"]
            GW["Full Sequence Attention<br/>(all positions)"]
        end

        L --> LocalAttn
        G --> GlobalAttn

        LocalAttn --> CAT["Concat Heads"]
        GlobalAttn --> CAT
        CAT --> OUT["Output Projection"]
        OUT --> O["(B, L, D)"]
    end
```

## Normalization Utilities

**File:** `src/prometheus/utils/norm.py`

```mermaid
flowchart LR
    subgraph Norm["Normalization Layers"]
        LN["LayerNorm<br/>- channels_last: F.layer_norm<br/>- channels_first: manual μ,σ"]
        GRN["GRN (Global Response Normalization)<br/>Gx = ‖x‖₂ over (H,W)<br/>Nx = Gx / mean(Gx)<br/>γ·(x·Nx) + β + x"]
    end
```

## Package Structure

```
src/prometheus/
├── __init__.py              # Package init, exports UNetTissue, DualUNet, all losses
├── config.py                # ModelConfig, TrainingConfig dataclasses
├── blocks/
│   ├── __init__.py
│   ├── attention.py         # LocalGlobalAttention
│   ├── convnext_block.py    # ConvNeXtBlock
│   ├── decoder_block.py     # DecoderBlock
│   ├── moe.py               # Expert, SparseMoE
│   └── transformer_block.py # EncoderTransformerBlock, EncoderTransformerStack
├── models/
│   ├── __init__.py
│   ├── _base_unet.py        # build_encoder, forward_encoder, build_decoder, forward_decoder
│   ├── unet_tissue.py       # UNetTissue
│   └── unet_dual.py         # DualUNet
├── losses/
│   ├── __init__.py
│   └── segmentation.py      # BCEWithLogitsLoss, DiceLoss, FocalLoss, CombinedLoss, MultiClassDiceLoss, MulticlassCombinedLoss, TverskyLoss
└── utils/
    ├── __init__.py
    └── norm.py              # LayerNorm, GRN
```

## Design Highlights

1. **ConvNeXt-v2 backbone:** Depthwise 7×7 conv + LayerNorm + GELU + GRN + DropPath — no BatchNorm, no ReLU.
2. **Local-Global Attention:** 50/50 head split between windowed (Swin-style) and full-sequence attention.
3. **Stop-gradient isolation:** `DualUNet` uses `.detach()` to prevent nuclei gradients from flowing into the tissue decoder, allowing independent training of each stream.
4. **Stochastic depth scheduling:** Drop path rates linearly increase from 0 to `drop_path_rate` across all blocks following ConvNeXt convention.