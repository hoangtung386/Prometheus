# Prometheus architecture

Prometheus uses dependency-oriented layers rather than organizing code around
individual experiments:

```text
domain <- data/models/metrics <- training/inference <- CLI/submission
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

```text
Image (3, H, W)
  │
  └─ Stem: Conv2d(3→96, k=4, s=4) + LayerNorm
  │
  ├─ Stage 1: ConvNeXtBlock ×3   dim=96   ↓2× downsample
  ├─ Stage 2: ConvNeXtBlock ×3   dim=192  ↓4×
  ├─ Stage 3: ConvNeXtBlock ×9   dim=384  ↓8×
  └─ Stage 4: ConvNeXtBlock ×3   dim=768  ↓16×
       │
       └─ Decoder (3 levels via DecoderBlock): upsampling + skip concat
             │
             └─ Output head: ConvTranspose2d(k=4,s=4) + Conv2d(→num_tissue_classes)
                    │
                    Tissue mask (C, H, W)
```

Each `ConvNeXtBlock`: DWConv 7×7 → LayerNorm → Linear(×4) → GELU → GRN → Linear(÷4) → residual + DropPath

Registered as `"tissue_convnext_unet"`, exported as `UNetTissue`.

### DualUNet — `models/multitask/dual_unet_legacy.py`

Legacy dual-stream model for simultaneous tissue + nuclei segmentation. **Not the target nuclei architecture** (kept for checkpoint compatibility):

```text
 Image (3, H, W)
   │
   ├── Tissue Stream ────────────────────────────────────────
   │   ConvNeXt Encoder → Decoder
   │   Output: tissue_mask (6, H, W) + full-res feature map (detached via .detach())
   │
   ├── Tissue→Nuclei Bridge
   │   TissueAttentionEncoder: downsamples detached feature map → flattened sequence
   │
   └── Nuclei Stream
       ConvNeXt Encoder → EncoderTransformerStack (6 blocks) → Decoder
       │
       Each transformer block:
         Self-Attn (LocalGlobalAttention, half heads local windowed,
                    half heads global full-sequence)
         → FFN → Cross-Attn (on tissue context) → SparseMoE (16 experts, top-2 gating)
         (all with Pre-LN + residual + dropout)
       │
       Output: nuclei_mask (11, H, W) + moe_auxiliary_loss
```

**Forward output:** 3-tuple `(tissue_logits, nuclei_logits, moe_loss)`.

Registered as `"legacy_dual_unet"`, exported as `DualUNet`.

---

## YOLO status

**YOLO is NOT implemented within Prometheus.** `YoloNucleiDetector` (`models/nuclei/yolo_adapter.py`) is a thin **inference-only adapter** that wraps a pre-trained `ultralytics.YOLO` model:

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

```text
PUMA dataset directory
  │
  └─ discover_puma_samples(root) — scans images/ + geojson_tissue/ + geojson_nuclei/
       │
       ├─ parse_tissue_geojson(path)  → list[(TissueClass, polygon)]
       ├─ parse_nuclei_geojson(path)  → list[NucleusInstance] (polygon, centroid, box)
       │
       ├─ rasterize_regions(...)      → mask (H, W) class indices
       ├─ rasterize_instances(...)    → instance mask
       │
       ├─ PumaTissueDataset           → (image, mask) for training UNetTissue
       ├─ PumaNucleiDataset           → (image, dict) for detection model training
       └─ PUMADataset (legacy)        → (image, {tissue, nuclei}) for DualUNet
            │
            └─ create_puma_dataloaders() → train/val DataLoaders + stratified splits
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

```text
predictions: list[Detection]      targets: list[Detection]
         │                                │
         └────── match_detections(radius_px=15.0, require_class_match=True) ──────┘
                        │
          MatchResult(matches, unmatched_pred, unmatched_target)
                        │
              nuclei_detection_metrics()
                        │
            Per-class precision, recall, F1
            macro_f1_summed (mean per-class F1)
            macro_f1_per_image (mean image-level macro F1)
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

---

## Model registry

`create_model(name, config)` (`models/registry.py`) — two registered factories:

| Name | Class |
|---|---|
| `"tissue_convnext_unet"` | `UNetTissue(config)` |
| `"legacy_dual_unet"` | `DualUNet(config)` |

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
migration phases and definition of done.
