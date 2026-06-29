# Prometheus architecture

Prometheus uses dependency-oriented layers rather than organizing code around
individual experiments:

```text
domain <- data/models/metrics <- training/inference <- CLI/submission
```

## Stable contracts

- `prometheus.domain`: canonical labels, geometry and typed predictions.
- `prometheus.data.puma`: PUMA filesystem discovery and strict GeoJSON parsing.
- `prometheus.metrics`: task metrics independent from file I/O.
- `prometheus.models`: complete models and optional framework adapters.

## Task boundaries

Tissue is semantic segmentation evaluated with Dice. Nuclei is instance
detection/classification evaluated by one-to-one centroid matching in a
15-pixel radius. The legacy `DualUNet` nuclei semantic head remains available
for comparison and checkpoint compatibility, but it is not the target nuclei
architecture.

## Compatibility

Existing imports under `prometheus.blocks`, `prometheus.utils`,
`prometheus.models.unet_*` and `prometheus.data.puma_dataset` remain valid during
the migration window. New code should use task model namespaces,
`prometheus.blocks` for reusable neural layers and `prometheus.data.puma`.

See [REFACTORING_GUIDE.md](../REFACTORING_GUIDE.md) for the complete file map,
migration phases and definition of done.
