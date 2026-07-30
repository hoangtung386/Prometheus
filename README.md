# Prometheus

Research framework for the [PUMA challenge](https://puma.grand-challenge.org/): tissue
segmentation and nuclei detection in advanced melanoma histopathology.

`PrometheusNet` shares a ConvNeXt-V2 encoder between two tasks, decodes tissue as semantic
segmentation, and detects nuclei as class-agnostic center-based instances from a
high-resolution feature pyramid.

## Status: read this first

The current model scores **52.90 official tissue micro Dice** on the preliminary test set,
against 78.23 for the winning entry and 55.48 for the challenge baseline. The gap is not
spread across the classes — it is two of them:

| Tissue class | Prometheus | Winner (TIAKong) | Runner-up (LSM) |
|---|---:|---:|---:|
| tumor | 89.19 | 93.58 | 92.07 |
| stroma | 81.00 | 83.59 | 81.28 |
| epidermis | **90.52** | 86.26 | 87.32 |
| necrosis | **0.00** | 82.04 | 46.79 |
| blood_vessel | **3.78** | 45.70 | 54.37 |
| **mean** | **52.90** | **78.23** | **72.37** |

Three of five classes are already at winning level. Bringing only `necrosis` and
`blood_vessel` to the winner's numbers, changing nothing else, would give **77.69**.

The diagnosis, the evidence behind it, and the ranked plan are in
[`docs/phan-tich-tissue-va-ke-hoach.md`](docs/phan-tich-tissue-va-ke-hoach.md) (Vietnamese).
Read section 4 before starting any experiment: it is a set of dataset audits that cost an
hour and decide which of the queued fixes actually matter.

New to the repo? Start with [`docs/handover.md`](docs/handover.md).

## Architecture

```text
domain       canonical taxonomy, geometry and framework-neutral types
data         PUMA discovery, GeoJSON parsing, rasterization, transforms, datasets
models       shared backbone, tissue/nuclei heads, fusion, typed outputs
losses       tissue, nuclei and multitask loss composition
metrics      official tissue micro Dice and 15-pixel centroid matching
engine       trainer, validation, EMA, schedule, checkpoint schema v2
inference    center decoding, dihedral TTA, source-space prediction
io           PUMA JSON/TIFF serializers
submission   output structure validation
config       config dataclasses + strict TOML loader
cli          audit, train, evaluate, predict, prepare-cellvit
api          stable composition root used by the CLI and the notebook
```

See [`docs/architecture.md`](docs/architecture.md) for the design decisions and the
contracts that must not be broken.

## Installation

```bash
uv sync --extra dev            # local development
uv sync --extra dev --extra viz  # plus matplotlib, for the notebook previews
pip install -r requirements.txt  # inside Colab
```

`pyproject.toml` is the dependency source of truth.

## Training

The supported workstation is the Colab notebook
[`notebooks/train.ipynb`](notebooks/train.ipynb): it verifies CUDA, audits every
annotation, persists the split next to the checkpoints, and resumes from `last.ckpt` after
a runtime disconnect.

For local or batch runs, use the CLI:

```bash
# Audit the dataset. Do this first: it reports label integrity, how much tissue area a
# naive rasterization would lose per class, how many images contain each class, and whether
# the images on disk match the 1024x1024 at 40x that the challenge test set uses.
uv run prometheus audit --data-root /path/to/puma

# Train from a reproducible TOML config
uv run prometheus train --config configs/experiment/baseline_multitask.toml

# Evaluate a checkpoint; --tta averages the eight dihedral views
uv run prometheus evaluate \
  --config configs/experiment/baseline_multitask.toml \
  --checkpoint runs/baseline_multitask_v2_transfer/best_tissue.ckpt --tta

# Produce the submission tissue TIFF and nuclei JSON
uv run prometheus predict \
  --config configs/experiment/baseline_multitask.toml \
  --checkpoint runs/baseline_multitask_v2_transfer/best_primary.ckpt \
  --input sample.tif --output predictions/sample

# Export nuclei polygons for CellViT-SAM-H classifier training
uv run prometheus prepare-cellvit \
  --data-root /path/to/puma --output /path/to/puma-cellvit \
  --cellvit-checkpoint /path/to/CellViT-SAM-H-x40-AMP.pth --run-dir /path/to/cellvit-runs
```

A run writes two checkpoints, because the challenge ranks the two tasks independently and
their best epochs differ: `best_primary.ckpt` (selected on
`config.evaluation.checkpoint_metric`) and `best_tissue.ckpt` (always the official tissue
micro Dice). `last.ckpt` exists for exact resume.

## Python API

```python
from prometheus.api import build_datamodule, build_model, build_trainer, load_config

config = load_config("configs/experiment/baseline_multitask.toml")
trainer = build_trainer(config, model=build_model(config, pretrained=True),
                        datamodule=build_datamodule(config))
trainer.fit()
```

Compose through `prometheus.api`. Importing internals directly couples you to layout that
is expected to move.

## Quality gates

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest -q
git diff --check
```

All four run in CI on Python 3.10 and 3.12. The test suite is CPU-only and never downloads
model weights. `uv run pre-commit install` wires the same checks into your commits.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the conventions these gates enforce.

## License

[MIT](LICENSE)
