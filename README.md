# Prometheus

Research framework for the PUMA melanoma histopathology challenge. PrometheusNet
shares shallow ConvNeXt features, decodes tissue semantically and detects nuclei
as center-based instances from a high-resolution feature pyramid.

See [architecture](docs/architecture.md) for design decisions
and implementation constraints.

## Current architecture

```text
domain       canonical labels, geometry and framework-neutral types
data         PUMA discovery, GeoJSON parsing, rasterization and datasets
models       shared backbone, tissue/nuclei heads, fusion and typed outputs
metrics      segmentation metrics and 15-pixel centroid matching
engine       typed trainer, exact evaluator and checkpoint schema v2
inference    center decoding and source-space prediction
io           PUMA JSON/TIFF serializers
submission   output structure validation
config       model and training config dataclasses + TOML loader
cli          audit, train, evaluate and predict commands
```

`DualUNet`, `UNetTissue` and semantic postprocessing remain available under
`prometheus.legacy.*` for old experiments. The production path uses
`PumaMultitaskDataset`, `PrometheusNet` and exact instance centroid F1.

See [architecture](docs/architecture.md) for architectural decisions,
migration constraints and the remaining research roadmap.

## Project structure

```text
prometheus/
├── configs/experiment/   TOML experiment configs
├── docs/                 design docs
├── notebooks/            Colab training notebook
├── src/prometheus/
│   ├── api.py            stable composition API
│   ├── blocks/           stable low-level convolutional layers
│   ├── cli/              CLI entry point
│   ├── config/           config dataclasses + TOML loader
│   ├── data/             datasets, transforms, PUMA IO
│   ├── domain/           canonical labels, geometry, types
│   ├── engine/           trainer, evaluator and checkpoint schema v2
│   ├── inference/        center decoder and source-space predictor
│   ├── io/               PUMA JSON/TIFF serializers
│   ├── legacy/           frozen semantic compatibility implementations
│   ├── losses/           tissue, nuclei and multitask losses
│   ├── metrics/          evaluation metrics
│   ├── models/           model architectures + registry
│   ├── nn/               stable neural-layer exports
│   ├── submission/       output validation
│   ├── training/         deprecated trainer compatibility package
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
pip install -r requirements.txt
```

`pyproject.toml` is the dependency source of truth.

## Commands (local / headless)

The CLI is useful for local runs or batch pipelines. Most users should use
[`notebooks/train.ipynb`](notebooks/train.ipynb) instead.

```bash
# Validate labels and annotation integrity
uv run prometheus audit --data-root /path/to/puma

# Train from a reproducible TOML config
uv run prometheus train --config configs/experiment/baseline_multitask.toml

# Evaluate a versioned checkpoint
uv run prometheus evaluate \
  --config configs/experiment/baseline_multitask.toml \
  --checkpoint runs/baseline_multitask_v1/best_primary.ckpt

# Produce tissue TIFF and nuclei JSON
uv run prometheus predict \
  --config configs/experiment/baseline_multitask.toml \
  --checkpoint runs/baseline_multitask_v1/best_primary.ckpt \
  --input sample.tif \
  --output predictions/sample
```

## Python API

```python
from prometheus.api import build_datamodule, build_model, build_trainer, load_config

config = load_config("configs/experiment/baseline_multitask.toml")
data = build_datamodule(config)
model = build_model(config)
trainer = build_trainer(config, model, data)
```

Legacy imports (`prometheus.legacy.DualUNet`, `prometheus.legacy.UNetTissue`)
are retained only for old experiments. New code should use `prometheus.api`
and `PrometheusNet`.

## Quality gates

```bash
uv run ruff check src tests
uv run pytest -q
git diff --check
```

The test suite is CPU-safe and does not download model weights.

## License

[MIT](LICENSE)
