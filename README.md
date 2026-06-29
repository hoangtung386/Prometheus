# Prometheus

Research framework for the PUMA melanoma histopathology challenge. Prometheus
separates tissue semantic segmentation, nuclei instance detection, official
evaluation, inference and submission serialization behind stable contracts.

See [architecture](docs/architecture.md) before development. The refactored
foundation is ready for team ownership, but YOLO training, official
evaluator-container parity and submission Docker validation remain explicit
backlog items.

## Current architecture

```text
domain       canonical labels, geometry and framework-neutral types
data         PUMA discovery, GeoJSON parsing, rasterization and datasets
models       tissue, nuclei adapters and legacy multitask models
metrics      segmentation metrics and 15-pixel centroid matching
training     legacy trainer plus versioned checkpoint services
inference    model-to-domain prediction pipeline and postprocessing
io           PUMA JSON/TIFF serializers
submission   output structure validation
config       model and training config dataclasses + TOML loader
cli          audit, train, evaluate and predict commands
```

`DualUNet` remains available as a legacy semantic baseline. New nuclei work
should use `PumaNucleiDataset` and a detection adapter such as
`YoloNucleiDetector`; model selection must be based on PUMA centroid F1 rather
than pixel Dice or detector mAP alone.

See [architecture](docs/architecture.md) for architectural decisions,
migration constraints and the remaining research roadmap.

## Project structure

```text
prometheus/
├── configs/experiment/   TOML experiment configs
├── docs/                 design docs
├── notebooks/            Colab training notebook
├── src/prometheus/
│   ├── blocks/           reusable neural layers
│   ├── cli/              CLI entry point
│   ├── config/           config dataclasses + TOML loader
│   ├── data/             datasets, transforms, PUMA IO
│   ├── domain/           canonical labels, geometry, types
│   ├── inference/        prediction pipeline
│   ├── io/               PUMA JSON/TIFF serializers
│   ├── losses/           segmentation loss functions
│   ├── metrics/          evaluation metrics
│   ├── models/           model architectures + registry
│   ├── nn/               extra neural ops
│   ├── submission/       output validation
│   ├── training/         trainer + checkpointing
│   ├── utils/            helpers (norm, etc.)
│   └── visualization/    plotting utilities
├── tests/                unit tests
├── pyproject.toml        build + deps
├── requirements.txt
└── README.md
```

## Usage

The primary entry point is the Colab notebook:

- [`notebooks/train.ipynb`](notebooks/train.ipynb) — full training pipeline
  on Google Colab (G4/A100) with dataset from Google Drive.

For local execution, the CLI is also available.

## Installation

Local development (recommended):

```bash
uv sync --extra dev
```

Optional visualization and YOLO support:

```bash
uv sync --extra dev --extra viz --extra yolo
```

Colab-compatible install (within the notebook):

```bash
pip install -e .
pip install tensorboard matplotlib
```

`pyproject.toml` is the dependency source of truth.

## Commands (local / headless)

The CLI is useful for local runs or batch pipelines. Most users should use
[`notebooks/train.ipynb`](notebooks/train.ipynb) instead.

```bash
# Validate labels and annotation integrity
uv run prometheus audit --data-root /path/to/puma

# Train from a reproducible TOML config
uv run prometheus train --config configs/experiment/legacy_dual.toml

# Evaluate a versioned checkpoint
uv run prometheus evaluate \
  --config configs/experiment/legacy_dual.toml \
  --checkpoint checkpoints/best.pt

# Produce tissue TIFF and nuclei JSON
uv run prometheus predict \
  --config configs/experiment/legacy_dual.toml \
  --checkpoint checkpoints/best.pt \
  --input sample.tif \
  --output predictions/sample
```

## Python API

```python
from prometheus.data import PumaNucleiDataset, PumaTissueDataset
from prometheus.domain import Detection, NucleusClass
from prometheus.metrics import nuclei_detection_metrics
from prometheus.models import create_model
```

Legacy imports (`DualUNet`, `UNetTissue`, `PUMADataset`) are retained during the
migration window to preserve checkpoints and existing callers.

## Quality gates

```bash
uv run ruff check src tests
uv run pytest -q
git diff --check
```

The test suite is CPU-safe and does not download model weights.

## License

[MIT](LICENSE)
