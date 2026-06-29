# Prometheus architecture

Prometheus uses dependency-oriented layers rather than organizing code around
individual experiments:

```mermaid
graph LR
    Domain[domain] --> Data[data / models / metrics]
    Data --> Training[training / inference]
    Training --> CLI[CLI / submission]
```

## Package layers

| Layer | Responsibility |
|---|---|
| `prometheus.domain` | Canonical labels (`TissueClass`, `NucleusClass`), geometry, and typed predictions (`PumaSample`, `Detection`, `NucleusInstance`) |
| `prometheus.data.puma` | PUMA filesystem discovery, strict GeoJSON parsing, rasterization, augmentations, `torch.utils.data.Dataset` classes |
| `prometheus.models` | Complete model architectures and optional framework adapters |
| `prometheus.metrics` | Task metrics: segmentation Dice/IoU + PUMA centroid-matching detection F1 |
| `prometheus.training` | Legacy `Trainer` + versioned `CheckpointService` |
| `prometheus.inference` | `PredictionPipeline` + `semantic_logits_to_detections` post-processing |
| `prometheus.io` | PUMA JSON/TIFF serializers |
| `prometheus.cli` | Audit, train, evaluate, and predict CLI commands |
| `prometheus.config` | `ModelConfig`, `TrainingConfig` dataclasses + TOML config loader |
| `prometheus.blocks` | Reusable neural layers (ConvNeXt, Transformer, MoE) |

---

## Working models

### UNetTissue / ConvNeXtUNet — `models/tissue/convnext_unet.py`

The primary tissue-segmentation model. A **pure ConvNeXt-V2 U-Net**:

```mermaid
graph TD
    Input["Image (3, H, W)"] --> Stem["Conv2d(3→96, k=4, s=4) + LayerNorm"]
    Stem --> S1["Stage 1: ConvNeXtBlock ×3<br/>dim=96  &nbsp;↓2×"]
    S1 --> S2["Stage 2: ConvNeXtBlock ×3<br/>dim=192 &nbsp;↓4×"]
    S2 --> S3["Stage 3: ConvNeXtBlock ×9<br/>dim=384 &nbsp;↓8×"]
    S3 --> S4["Stage 4: ConvNeXtBlock ×3<br/>dim=768 &nbsp;↓16×"]
    S4 --> Dec["Decoder (3 levels DecoderBlock)<br/>upsample + skip concat"]
    S3 -. "skip" .-> Dec
    S2 -. "skip" .-> Dec
    S1 -. "skip" .-> Dec
    Dec --> Head["Output head<br/>ConvTranspose2d(k=4,s=4) + Conv2d(→C)"]
    Head --> Out["Tissue mask (C, H, W)"]
```

Each `ConvNeXtBlock`: DWConv 7×7 → LayerNorm → Linear(×4) → GELU → GRN → Linear(÷4) → residual + DropPath

Registered as `"tissue_convnext_unet"`, exported as `UNetTissue`.

### DualUNet — `models/multitask/dual_unet_legacy.py`

Legacy dual-stream model for simultaneous tissue + nuclei segmentation. **Not the target nuclei architecture** (kept for checkpoint compatibility):

```mermaid
graph TD
    Input["Image (3, H, W)"] --> TissueEnc["ConvNeXt Encoder"]
    Input --> NucEnc["ConvNeXt Encoder"]

    subgraph Tissue["Tissue Stream"]
        TissueEnc --> TissueDec["Decoder"]
        TissueDec --> TMask["tissue_mask (6, H, W)"]
        TissueDec --> FeatMap["full-res feature map"]
    end

    subgraph Bridge["Tissue→Nuclei Bridge"]
        FeatMap -. "detach()" .-> TAE["TissueAttentionEncoder<br/>downsample → flatten"]
    end

    subgraph Nuclei["Nuclei Stream"]
        NucEnc --> XFMR["EncoderTransformerStack (6 blocks)"]
        TAE -. "cross-attn context" .-> XFMR
        XFMR --> NucDec["Decoder"]
        NucDec --> NMask["nuclei_mask (11, H, W)"]
        XFMR --> MoELoss["moe_auxiliary_loss"]
    end

    subgraph TransformerBlock["Each transformer block"]
        direction TB
        SA["Self-Attn<br/>LocalGlobalAttention<br/>½ heads local · ½ heads global"] --> FFN["FFN"]
        FFN --> CA["Cross-Attn<br/>(on tissue context)"]
        CA --> MoE["SparseMoE<br/>16 experts · top-2 gating"]
    end
```

**Forward output:** 3-tuple `(tissue_logits, nuclei_logits, moe_loss)`.

Registered as `"legacy_dual_unet"`, exported as `DualUNet`.

---

## YOLO status

**YOLO is NOT implemented within Prometheus.** `YoloNucleiDetector` (`models/nuclei/yolo_adapter.py`) is a thin **inference-only adapter** that wraps a pre-trained `ultralytics.YOLO` model:

```mermaid
graph LR
    subgraph External["External (not in Prometheus)"]
        Pretrained["Pre-trained YOLO weights.pt"]
    end

    subgraph Prometheus["Prometheus codebase"]
        Adapter["YoloNucleiDetector(model_or_weights)<br/>models/nuclei/yolo_adapter.py"]
        Input["images: torch.Tensor"] --> Adapter
        Pretrained -. "loads weights" .-> Adapter
        Adapter --> Predict["predict()<br/>→ ultralytics.YOLO.predict()"]
        Predict --> Results["list[Detection]<br/>(centroid, label, confidence, box_xyxy)"]
    end

    subgraph NotBuilt["Not built"]
        Note1["✗ No training loop"]
        Note2["✗ No YOLO loss"]
        Note3["✗ No dataset collator"]
        Note4["✗ Not in model registry"]
        Note5["✗ Not usable via PredictionPipeline"]
    end

    Predict -.-> Note1
```

- Constructor loads external weights via `ultralytics.YOLO(weights)`.
- `predict(images)` runs inference and converts outputs to `Detection` dataclasses (centroid, label, confidence, box).
- **Not registered** in the model registry.
- **Not usable** through `PredictionPipeline` (which expects DualUNet's 3-tuple output).
- **No YOLO training loop**, no YOLO loss, no YOLO dataset collator in Prometheus.
- Requires optional `[yolo]` extra (`uv sync --extra yolo`) for the `ultralytics` dependency.

YOLO training is explicitly documented as **backlog** (see `README.md` and `REFACTORING_GUIDE.md`).

---

## Building blocks (`prometheus.blocks`)

| Block | File | Purpose |
|---|---|---|
| `ConvNeXtBlock` | `blocks/convnext_block.py` | ConvNeXt V2: DWConv 7×7 → LN → Linear(×4) → GELU → GRN → Linear(÷4) → DropPath |
| `DecoderBlock` | `blocks/decoder_block.py` | Upsample (ConvTranspose2d) → skip concat + 1×1 proj → ConvNeXt body |
| `LocalGlobalAttention` | `blocks/attention.py` | Multi-head: 50/50 split local (windowed, Swin-style) / global (full-sequence). Supports self- and cross-attention |
| `EncoderTransformerBlock` | `blocks/transformer_block.py` | Pre-LN: Self-Attn → FFN → Cross-Attn (optional) → SparseMoE |
| `EncoderTransformerStack` | `blocks/transformer_block.py` | Stack of N `EncoderTransformerBlock`s, accumulates MoE auxiliary loss |
| `Expert` | `blocks/moe.py` | MLP: `Linear(d_expert → d_ff×4) → SiLU → Linear(d_ff×4 → d_expert)` |
| `SparseMoE` | `blocks/moe.py` | Down-proj → top-2 gating → 16 experts (weighted sum) → Up-proj + load-balancing loss |
| `LayerNorm` | `utils/norm.py` | Custom LN: supports `channels_last` (F.layer_norm) and `channels_first` (manual µ,σ) |
| `GRN` | `utils/norm.py` | Global Response Normalization: `Gx = ‖x‖₂`, `Nx = Gx / mean(Gx)`, `out = γ·(x·Nx) + β + x` |

---

## Data pipeline

```mermaid
graph TD
    Root["PUMA dataset directory"] --> Disc["discover_puma_samples(root)"]

    Disc --> TissueGJ["parse_tissue_geojson(path)<br/>→ list(TissueClass, polygon)"]
    Disc --> NucGJ["parse_nuclei_geojson(path)<br/>→ list(NucleusInstance)"]

    TissueGJ --> RastReg["rasterize_regions()<br/>→ mask (H, W) class indices"]
    NucGJ --> RastInst["rasterize_instances()<br/>→ instance mask"]

    RastReg --> TissueDS["PumaTissueDataset<br/>→ (image, mask) for UNetTissue"]
    NucGJ --> NucDS["PumaNucleiDataset<br/>→ (image, dict) for detection models"]
    RastReg --> LegacyDS["PUMADataset (legacy)<br/>→ (image, {tissue,nuclei}) for DualUNet"]
    RastInst --> LegacyDS

    TissueDS --> DL["create_puma_dataloaders()<br/>train/val DataLoaders<br/>+ stratified splits"]
    NucDS --> DetDL["collate_detection()<br/>→ detection DataLoader"]
    LegacyDS --> DL

    NucDS -. "future" .-> YOLO["YoloNucleiDetector<br/>(inference-only adapter)"]
    YOLO -. "needs training loop" .-> Backlog["BACKLOG: YOLO training"]
```

**Transforms:**
- Geometric: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotate90`, `ElasticDeformation`
- Photometric: `Normalize`, `NormalizeTile` (per-channel percentile + z-score), `RandomBrightnessContrast`, `RandomChannelJitter`, `RandomGamma`, `RandomGaussianNoise`
- Pre-built: `train_transform()` (all augs + NormalizeTile), `val_transform()` / `test_transform()` (NormalizeTile only)

---

## Evaluation pipeline

### Tissue — semantic segmentation

`SegmentationEvaluator` (`metrics/evaluator.py`):
- Accumulates per-class TP/FP/FN/TN across batches.
- `compute()` → `dice`, `iou`, `sensitivity`, `precision`, `specificity`, `accuracy`.
- Foreground-only mean (skips background class 0).

### Nuclei — instance detection

```mermaid
graph LR
    subgraph Models["Prediction sources"]
        DL["DualUNet nuclei head<br/>semantic_logits_to_detections()"]
        YOLO["YoloNucleiDetector<br/>(inference adapter)"]
    end

    DL --> Pred["predictions<br/>list[Detection]"]
    YOLO --> Pred

    Target["targets<br/>list[Detection]"] --> Match["match_detections(radius_px=15.0)"]
    Pred --> Match

    Match --> Result["MatchResult<br/>(matches, unmatched_pred, unmatched_target)"]
    Result --> Metrics["nuclei_detection_metrics()"]

    Metrics --> F1["Per-class precision, recall, F1"]
    Metrics --> F1sum["macro_f1_summed<br/>(mean per-class F1)"]
    Metrics --> F1img["macro_f1_per_image<br/>(mean image-level macro F1)"]
```

`match_detections` implements PUMA-official centroid matching: for each target, finds the best unused prediction within 15 pixels that matches the class. Tie-breaking by highest confidence, then nearest distance.

---

## Inference pipeline

`PredictionPipeline` (`inference/pipeline.py`) — currently hardcoded for `DualUNet`:
1. Runs `model(images.to(device))` → expects 3-tuple `(tissue_logits, nuclei_logits, _)`.
2. `tissue_mask = tissue_logits.argmax(dim=1)`.
3. `semantic_logits_to_detections(nuclei_logits)`:
   - softmax → argmax for class mask.
   - `cv2.connectedComponentsWithStats` per foreground class → centroid, mean probability.
   - Returns `list[list[Detection]]`.

`YoloNucleiDetector` bypasses the pipeline entirely — call `.predict(images)` directly for a `list[list[Detection]]` from an external YOLO model.

---

## Model registry

`create_model(name, config)` (`models/registry.py`) — two registered factories:

| Name | Class | Status |
|---|---|---|
| `"tissue_convnext_unet"` | `UNetTissue(config)` | ✅ Working |
| `"legacy_dual_unet"` | `DualUNet(config)` | ✅ Working |
| `"yolo_nuclei"` | — | ❌ Not registered (inference adapter only) |

---

## Task boundaries

Tissue is **semantic segmentation** evaluated with Dice. Nuclei is **instance detection/classification** evaluated by one-to-one centroid matching in a 15-pixel radius. The legacy `DualUNet` nuclei semantic head remains available for comparison and checkpoint compatibility, but it is not the target nuclei architecture.

---

## Configuration

`ModelConfig` defaults (`config/schemas.py`):

| Field | Default | Description |
|---|---|---|
| `in_chans` | 3 | Input channels |
| `num_tissue_classes` | 6 | Tissue output classes |
| `num_nuclei_classes` | 11 | Nuclei output classes |
| `encoder_dims` | [96, 192, 384, 768] | ConvNeXt stage dimensions |
| `encoder_depths` | [3, 3, 9, 3] | Blocks per stage |
| `drop_path_rate` | 0.1 | Stochastic depth |
| `n_heads` | 8 | Attention heads |
| `d_ff` | 3072 | Transformer FFN dimension |
| `d_expert` | 256 | MoE expert hidden dim |
| `window_size` | 8 | Local attention window |
| `num_transformer_blocks` | 6 | Transformer stack depth |
| `num_experts` | 16 | MoE expert count |
| `moe_top_k` | 2 | Experts per token |
| `use_tissue_context` | True | Cross-attention to tissue |

---

## Compatibility

Existing imports under `prometheus.blocks`, `prometheus.utils`,
`prometheus.models.unet_*` and `prometheus.data.puma_dataset` remain valid during
the migration window. New code should use task model namespaces,
`prometheus.blocks` for reusable neural layers and `prometheus.data.puma`.

See [REFACTORING_GUIDE.md](../REFACTORING_GUIDE.md) for the complete file map,
migration phases and definition of done. _(Removed — refer to `docs/architecture.md`,
`README.md` and git history instead.)_
