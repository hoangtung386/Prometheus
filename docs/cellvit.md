# CellViT-SAM-H nuclei pipeline

> **Priority note.** The nuclei branch already scores near the Track-2 runner-up
> (21.72 macro F1 against KongNet's 26.56), while tissue sits below the challenge baseline.
> This path is documented and working, but it is not where the remaining points are. See
> [`phan-tich-tissue-va-ke-hoach.md`](phan-tich-tissue-va-ke-hoach.md) section 2.4.

The production nuclei path uses the official CellViT++ transfer boundary:

1. `CellViT-SAM-H-x40-AMP.pth` detects and segments cells at 0.25 MPP.
2. The detector stays frozen and emits one 1280-dimensional token per cell.
3. A small classifier learns the ten PUMA Track-2 nucleus types.

This is intentionally separate from `PrometheusNet`. CellViT++ has its own HV-map
postprocessing and checkpoint schema; copying SAM-H tensors into the CenterPoint head
would not transfer the learned detector.

## Prepare data

Download `CellViT-SAM-H-x40-AMP.pth` from the official CellViT++ checkpoint folder,
then export PUMA polygons to instance and type maps:

```bash
prometheus prepare-cellvit \
  --data-root /path/to/dataset_PUMA \
  --output /path/to/puma-cellvit \
  --cellvit-checkpoint /path/to/CellViT-SAM-H-x40-AMP.pth \
  --run-dir /path/to/runs/cellvit-puma-track2
```

The exporter preserves every polygon as an instance and uses the official PUMA order:
endothelium, plasma cell, stroma, tumor, histiocyte, apoptosis, epithelium,
melanophage, neutrophil, lymphocyte. It writes the `SegmentationDataset` layout,
train/validation file lists, and `cellvit-puma-track2.yaml`.

## Train only the PUMA classifier

Use a Python 3.10 CellViT++ environment as required by the upstream project:

```bash
git clone https://github.com/TIO-IKIM/CellViT-plus-plus.git
bash scripts/train_cellvit_classifier.sh \
  /path/to/CellViT-plus-plus \
  /path/to/puma-cellvit/cellvit-puma-track2.yaml
```

Do not fine-tune SAM-H first. With roughly 205 PUMA ROIs, begin with the detector
frozen and train the classifier. Unfreeze only the final decoder blocks in a later
controlled experiment if out-of-fold detection recall is demonstrably poor.

The upstream license and the licenses of its pretrained weights must be reviewed
before redistribution or commercial use.
