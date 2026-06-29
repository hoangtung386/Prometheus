# Prometheus

Research framework for the PUMA melanoma histopathology challenge. Prometheus
separates tissue semantic segmentation, nuclei instance detection, official
evaluation, inference and submission serialization behind stable contracts.

Read [HANDOVER.md](HANDOVER.md) before development. The refactored foundation
is ready for team ownership, but YOLO training, official evaluator-container
parity and submission Docker validation remain explicit backlog items.

## Current architecture

```text
domain       canonical labels, geometry and framework-neutral types
data/puma    discovery, strict GeoJSON parsing, rasterization and datasets
models       tissue, nuclei adapters and legacy multitask models
metrics      segmentation metrics and 15-pixel centroid matching
training     legacy trainer plus versioned checkpoint services
inference    model-to-domain prediction pipeline and postprocessing
io           PUMA JSON/TIFF serializers
cli          audit, train, evaluate and predict commands
```

`DualUNet` remains available as a legacy semantic baseline. New nuclei work
should use `PumaNucleiDataset` and a detection adapter such as
`YoloNucleiDetector`; model selection must be based on PUMA centroid F1 rather
than pixel Dice or detector mAP alone.

See [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) for architectural decisions,
migration constraints and the remaining research roadmap.

## Installation

```bash
uv sync --extra dev
```

Optional visualization and YOLO support:

```bash
uv sync --extra dev --extra viz --extra yolo
```

`pyproject.toml` is the dependency source of truth.

## Commands

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
