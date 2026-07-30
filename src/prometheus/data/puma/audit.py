"""Dataset audits that answer the questions a run cannot answer for itself.

Three checks, all cheap, all worth running before any training experiment. They exist
because a silent data defect and a genuine modelling limitation look identical in the
metrics, and this project already lost a full 5-fold run to the difference:

:func:`audit_puma_dataset`
    Labels parse, every annotation is readable, and the class inventory is what you expect.

:func:`audit_tissue_rasterization`
    How much tissue area the mask *would* lose to dropped interior rings and to arbitrary
    paint order. If a rare class shows a large delta here, the training masks are wrong and
    no amount of training will recover the class.

:func:`audit_resolution`
    Whether the images on disk are the native 1024x1024 at 40x that the challenge test set
    uses, and whether the annotation coordinates agree with them. Training on downsampled
    regions of interest and submitting on native ones is a silent domain shift that
    destroys the thin classes first.

See ``docs/phan-tich-tissue-va-ke-hoach.md`` section 4.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ...domain import TISSUE_CLASS_NAMES, TISSUE_CLASS_TO_INDEX, TissueClass
from ...domain.geometry import Polygon
from .discovery import discover_puma_samples
from .geojson import parse_nuclei_geojson, parse_tissue_geojson
from .rasterize import TISSUE_PAINT_PRIORITY, rasterize_tissue_regions

__all__ = [
    "audit_puma_dataset",
    "audit_resolution",
    "audit_tissue_rasterization",
]

PUMA_NATIVE_SIZE = (1024, 1024)
"""Native region-of-interest size of the PUMA release, at 40x (~0.22 um/pixel)."""


def audit_puma_dataset(root: str | Path) -> dict[str, object]:
    """Verify every annotation parses and report the class inventory."""
    samples = discover_puma_samples(root)
    tissue_counts: Counter[str] = Counter()
    nuclei_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for sample in samples:
        try:
            for label, _ in parse_tissue_geojson(sample.tissue_annotation_path):
                tissue_counts[label.value] += 1
            for instance in parse_nuclei_geojson(sample.nuclei_annotation_path):
                nuclei_counts[instance.label.value] += 1
        except (OSError, TypeError, ValueError) as error:  # noqa: PERF203
            errors.append({"sample_id": sample.sample_id, "error": str(error)})
    return {
        "sample_count": len(samples),
        "tissue_region_counts": dict(sorted(tissue_counts.items())),
        "nuclei_instance_counts": dict(sorted(nuclei_counts.items())),
        "errors": errors,
    }


def _rasterize_naive(
    regions: list[tuple[TissueClass, Polygon]],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Rasterize the way a naive implementation would: file order, interior rings dropped.

    Kept only as the audit's baseline, so the cost of both defects is measurable rather
    than argued about.
    """
    stripped = [(label, Polygon(exterior=polygon.exterior)) for label, polygon in regions]
    order = tuple(dict.fromkeys(label for label, _ in stripped))
    missing = tuple(item for item in TISSUE_PAINT_PRIORITY if item not in order)
    return rasterize_tissue_regions(stripped, image_size, priority=missing + order)


def _class_report(
    name: str,
    naive_pixels: Counter[str],
    correct_pixels: Counter[str],
    images_containing: Counter[str],
) -> dict[str, object]:
    """Summarise one class, keeping "the naive mask had none of this" distinguishable.

    A percentage change is meaningless against a zero baseline, so it is reported as ``None``
    and the ``absent_from_naive_mask`` flag carries the finding instead. That flag being true
    for a rare class is the strongest signal in the whole audit: the class was not merely
    under-represented in the old training masks, it was absent.
    """
    naive, correct = naive_pixels[name], correct_pixels[name]
    return {
        "pixels_naive": naive,
        "pixels_correct": correct,
        "pixel_delta_percent": round(100.0 * (correct - naive) / naive, 2) if naive else None,
        "absent_from_naive_mask": naive == 0 and correct > 0,
        "images_containing": images_containing[name],
    }


def audit_tissue_rasterization(
    root: str | Path,
    image_size: tuple[int, int] = PUMA_NATIVE_SIZE,
) -> dict[str, object]:
    """Measure what naive tissue rasterization would lose, per class.

    Args:
        root: PUMA dataset root.
        image_size: Rasterization size; use the native annotation resolution.

    Returns:
        ``interior_ring_count`` (dropping these is defect 1),
        ``images_with_interior_rings``,
        ``pixels_naive`` / ``pixels_correct`` / ``pixel_delta_percent`` /
        ``absent_from_naive_mask`` per class, and ``images_containing`` per class. The last
        one sizes rare-class oversampling and tells you whether a class can be stratified
        across folds at all.
    """
    samples = discover_puma_samples(root)
    naive_pixels: Counter[str] = Counter()
    correct_pixels: Counter[str] = Counter()
    images_containing: Counter[str] = Counter()
    interior_rings = 0
    images_with_interior_rings = 0

    for sample in samples:
        regions = parse_tissue_geojson(sample.tissue_annotation_path)
        rings = sum(len(polygon.holes) for _, polygon in regions)
        interior_rings += rings
        images_with_interior_rings += int(rings > 0)

        naive = _rasterize_naive(regions, image_size)
        correct = rasterize_tissue_regions(regions, image_size)
        for name in TISSUE_CLASS_NAMES:
            index = TISSUE_CLASS_TO_INDEX[name]
            naive_pixels[name] += int((naive == index).sum())
            present = int((correct == index).sum())
            correct_pixels[name] += present
            images_containing[name] += int(present > 0)

    per_class = {
        name: _class_report(name, naive_pixels, correct_pixels, images_containing) for name in TISSUE_CLASS_NAMES
    }
    return {
        "sample_count": len(samples),
        "image_size": list(image_size),
        "interior_ring_count": interior_rings,
        "images_with_interior_rings": images_with_interior_rings,
        "paint_priority": [item.value for item in TISSUE_PAINT_PRIORITY],
        "per_class": per_class,
    }


def audit_resolution(root: str | Path) -> dict[str, object]:
    """Compare image sizes on disk against the annotation coordinate range.

    A ``coordinate_max`` of about twice ``image_size`` means the annotations were never
    rescaled alongside a downsampled image, so every mask is misaligned. A native size
    other than 1024x1024 means training and the challenge test set disagree on
    magnification.
    """
    import tifffile  # noqa: PLC0415 - only this audit needs the TIFF header reader

    samples = discover_puma_samples(root)
    size_counts: Counter[tuple[int, int]] = Counter()
    coordinate_maxima: list[float] = []
    for sample in samples:
        with tifffile.TiffFile(sample.image_path) as handle:
            height, width = handle.pages[0].shape[:2]
        size_counts[(int(height), int(width))] += 1
        maximum = 0.0
        for _, polygon in parse_tissue_geojson(sample.tissue_annotation_path):
            maximum = max(maximum, float(polygon.exterior.max(initial=0.0)))
        coordinate_maxima.append(maximum)

    sizes = {f"{height}x{width}": count for (height, width), count in sorted(size_counts.items())}
    native = list(size_counts) == [PUMA_NATIVE_SIZE]
    largest = max(coordinate_maxima) if coordinate_maxima else 0.0
    shortest_side = min(min(size) for size in size_counts) if size_counts else 0
    return {
        "sample_count": len(samples),
        "image_sizes": sizes,
        "matches_puma_native_size": native,
        "coordinate_max": round(largest, 1),
        "coordinates_exceed_image": bool(shortest_side and largest > shortest_side + 1),
    }
