"""Dataset integrity audit service used by CLI and tests."""

from __future__ import annotations

from collections import Counter

from .discovery import discover_puma_samples
from .geojson import parse_nuclei_geojson, parse_tissue_geojson


def audit_puma_dataset(root) -> dict[str, object]:
    samples = discover_puma_samples(root)
    tissue_counts: Counter[str] = Counter()
    nuclei_counts: Counter[str] = Counter()
    errors = []
    for sample in samples:
        try:
            for label, _ in parse_tissue_geojson(sample.tissue_annotation_path):
                tissue_counts[label.value] += 1
            for instance in parse_nuclei_geojson(sample.nuclei_annotation_path):
                nuclei_counts[instance.label.value] += 1
        except (OSError, ValueError) as error:
            errors.append({"sample_id": sample.sample_id, "error": str(error)})
    return {
        "sample_count": len(samples),
        "tissue_region_counts": dict(sorted(tissue_counts.items())),
        "nuclei_instance_counts": dict(sorted(nuclei_counts.items())),
        "errors": errors,
    }
