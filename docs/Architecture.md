# Prometheus Architecture

## Overview

Prometheus is a medical image segmentation framework built on a **ConvNeXt-v2 U-Net** backbone. It provides three model variants:

| Model | Description |
|-------|-------------|
| **UNetTissue** | Standard ConvNeXt U-Net for tissue segmentation |
| **UNetNuclei** | Extended U-Net with MinkowskiEngine sparse tissue-mask encoder + transformer bottleneck |
| **DualUNet** | Dual-stream architecture: tissue stream + nuclei stream with cross-attention fusion |

## Core Configuration

```python
ModelConfig:
  in_chans: 3
  num_classes: 1
  encoder_dims: [96, 192, 384, 768]
  encoder_depths: [3, 3, 9, 3]
  drop_path_rate: 0.1
  D: 2  # spatial dimension (2D or 3D for MinkowskiEngine)
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
        +q_proj, k_proj, v_proj: Linear
        +out_proj: Linear
        +forward(x, context?) Tensor
    }

    class MinkowskiConvNeXtBlock {
        +dwconv: MinkowskiConvolution(3x3, groups=dim)
        +norm: MinkowskiLayerNorm
        +pwconv1: MinkowskiLinear(dim, 4*dim)
        +act: GELU
        +grn: MinkowskiGRN(4*dim)
        +pwconv2: MinkowskiLinear(4*dim, dim)
        +drop_path: MinkowskiDropPath
        +forward(x) SparseTensor
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
        L0["Level 0<br/>(B, 384, H/16, W/16)<br/>Upsample + Skip + ConvNeXtBlock ×3"]
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

## UNetNuclei

**File:** `src/prometheus/models/unet_nuclei.py`

Extends the base U-Net with a **MinkowskiEngine sparse tissue-mask encoder** and a concatenated second encoder at the bottleneck. Requires the tissue mask as a second input.

```mermaid
flowchart TB
    subgraph Inputs["Inputs"]
        I["Image<br/>(B, 3, H, W)"]
        TM["Tissue Mask<br/>(B, N_points)"]
    end

    subgraph Encoder["EncoderNuclei<br/>(ConvNeXt U-Net Encoder)"]
        ENC["Stem + 4 Stages<br/>with Downsampling<br/>produces bottleneck + skips"]
    end

    subgraph Sparse["EncoderFeaturesMaskTissue<br/>(MinkowskiEngine)"]
        UM["Upsample Mask<br/>to (H/16, W/16)"]
        MASK["Mask Features<br/>x *= (1 - mask)"]
        SP["to_sparse() → SparseTensor"]
        MS1["Stage 1<br/>MinkowskiConvNeXtBlock ×3<br/>dim=96"]
        MD1["Down 1<br/>MinkowskiConv<br/>2x2, s2"]
        MS2["Stage 2<br/>MinkowskiConvNeXtBlock ×3<br/>dim=192"]
        MD2["Down 2<br/>MinkowskiConv<br/>2x2, s2"]
        MS3["Stage 3<br/>MinkowskiConvNeXtBlock ×9<br/>dim=384"]
        MD3["Down 3<br/>MinkowskiConv<br/>2x2, s2"]
        MS4["Stage 4<br/>MinkowskiConvNeXtBlock ×3<br/>dim=768"]
        DENSE["dense() → Tensor<br/>(B, 768, H/32, W/32)"]
    end

    subgraph Fusion["Feature Fusion"]
        ADD["x + tissue_features"]
    end

    subgraph TransformerEncoder["TransformerEncoderNuclei<br/>(Second ConvNeXt Encoder)"]
        TE["Stem + 4 Stages<br/>with Downsampling"]
    end

    subgraph Decoder["DecoderNuclei<br/>(ConvNeXt U-Net Decoder)"]
        DEC["3 Levels + Output Head"]
    end

    I --> ENC
    ENC -->|"bottleneck (B, 768, H/32, W/32)"| ADD
    ENC -->|"skips"| Decoder

    TM --> UM
    ENC -->|"bottleneck features"| UM
    UM --> MASK
    MASK --> SP
    SP --> MS1
    MS1 --> MD1
    MD1 --> MS2
    MS2 --> MD2
    MD2 --> MS3
    MS3 --> MD3
    MD3 --> MS4
    MS4 --> DENSE
    DENSE --> ADD

    ADD --> TE
    TE -->|"bottleneck"| Decoder
    TE -->|"skips"| Decoder
    DEC --> O["Output<br/>(B, num_classes, H, W)"]

    style Inputs fill:#e1f5fe
    style Encoder fill:#e8f5e9
    style Sparse fill:#f3e5f5
    style Fusion fill:#fff8e1
    style TransformerEncoder fill:#e0f2f1
    style Decoder fill:#fff3e0
```

### MinkowskiConvNeXtBlock Detail

```mermaid
flowchart LR
    subgraph MinkowskiBlock["MinkowskiConvNeXtBlock(dim, drop_path, D)"]
        I["SparseTensor<br/>(features: C)"] --> DW["MinkowskiConvolution<br/>3×3 depthwise, groups=C"]
        DW --> LN["MinkowskiLayerNorm"]
        LN --> FC1["MinkowskiLinear(C → 4C)"]
        FC1 --> G["GELU"]
        G --> GRN["MinkowskiGRN(4C)"]
        GRN --> FC2["MinkowskiLinear(4C → C)"]
        FC2 --> DP["MinkowskiDropPath"]
        DP --> ADD["+ Residual"]
        ADD --> O["SparseTensor<br/>(features: C)"]
    end
```

## DualUNet

**File:** `src/prometheus/models/unet_dual.py`

A dual-stream architecture with **stop-gradient** isolation between tissue and nuclei streams. The tissue stream's feature map is encoded and fused into the nuclei bottleneck via **LocalGlobalAttention** cross-attention. Does not require MinkowskiEngine.

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
        TAE["TissueAttentionEncoder<br/>Stem + 3 Downsamples<br/>→ (B, 768, H/32, W/32)"]
    end

    subgraph Nuclei["Nuclei Stream"]
        direction TB
        NE["Encoder<br/>Stem + 4 ConvNeXt Stages"]
        CA["LocalGlobalAttention<br/>d_model=768, n_heads=8<br/>window_size=2<br/>Q: nuclei, K/V: tissue"]
        ND["TissueDecoder<br/>(nuclei decoder)<br/>3 Decoder Levels"]
        OH2["Mask Head<br/>ConvTranspose2d(4x4) + Conv2d(1x1)"]
    end

    I --> TE
    TE -->|"skips"| TD
    TE -->|"bottleneck (B, 768, H/32, W/32)"| TD
    TD --> FH
    FH --> OH1
    OH1 --> TM["Tissue Mask"]

    FH --> SG
    SG --> TAE
    TAE --> CA

    I --> NE
    NE -->|"bottleneck (B, 768, H/32, W/32)"| CA
    NE -->|"skips"| ND
    CA --> ND
    ND --> OH2
    OH2 --> NM["Nuclei Mask"]

    style Input fill:#e1f5fe
    style Tissue fill:#e8f5e9
    style Bridge fill:#fff8e1
    style Nuclei fill:#fce4ec
```

### Cross-Attention Fusion Detail

```mermaid
flowchart LR
    subgraph Fusion["Cross-Attention (in DualUNet)"]
        NB["Nuclei Bottleneck<br/>(B, 768, H/32, W/32)"] --> FL["Flatten + Transpose<br/>(B, L, 768)"]
        TB["Tissue Bottleneck<br/>(B, 768, H/32, W/32)"] --> FL2["Flatten + Transpose<br/>(B, L, 768)"]

        FL --> Q["Q Projection"]
        FL2 --> K["K Projection"]
        FL2 --> V["V Projection"]

        Q --> SPLIT["Split Heads<br/>(local: 4, global: 4)"]
        K --> SPLIT2["Split Heads<br/>(local: 4, global: 4)"]
        V --> SPLIT3["Split Heads<br/>(local: 4, global: 4)"]

        subgraph Global["Global Attention"]
            GLO["Softmax(Q_glob @ K_glob^T / √d) @ V_glob"]
        end

        subgraph Local["Windowed Attention"]
            LO["Window reshape (W=2)<br/>Softmax(Q_loc @ K_loc^T / √d) @ V_loc"]
        end

        SPLIT --> Global
        SPLIT --> Local
        SPLIT2 --> Global
        SPLIT2 --> Local
        SPLIT3 --> Global
        SPLIT3 --> Local

        Global --> CAT["Concat Heads"]
        Local --> CAT
        CAT --> O["Output Projection"]
        O --> RESHAPE["Reshape → (B, 768, H/32, W/32)"]
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
        V --> SPLIT3["Split Heads"]

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
├── config.py                     ModelConfig dataclass
├── blocks/
│   ├── convnext_block.py         ConvNeXtBlock
│   ├── decoder_block.py          DecoderBlock
│   ├── attention.py              LocalGlobalAttention, CrossAttention
│   └── minkowski_block.py        MinkowskiConvNeXtBlock
├── models/
│   ├── _base_unet.py             build_encoder, forward_encoder,
│   │                             build_decoder, forward_decoder
│   ├── unet_tissue.py            UNetTissue
│   ├── unet_nuclei.py            UNetNuclei
│   └── unet_dual.py              DualUNet
└── utils/
    ├── norm.py                   LayerNorm, GRN
    └── minkowski_utils.py        MinkowskiGRN, MinkowskiDropPath,
                                  MinkowskiLayerNorm, require_minkowski_engine
```

## Design Highlights

1. **ConvNeXt-v2 backbone:** Depthwise 7×7 conv + LayerNorm + GELU + GRN + DropPath — no BatchNorm, no ReLU.
2. **Local-Global Attention:** 50/50 head split between windowed (Swin-style) and full-sequence attention.
3. **MinkowskiEngine sparse conv:** `UNetNuclei` uses sparse 3×3 ConvNeXt blocks on masked tissue features for efficiency.
4. **Stop-gradient isolation:** `DualUNet` uses `.detach()` to prevent nuclei gradients from flowing into the tissue decoder, allowing independent training of each stream.
5. **Stochastic depth scheduling:** Drop path rates linearly increase from 0 to `drop_path_rate` across all blocks following ConvNeXt convention.
