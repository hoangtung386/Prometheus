"""Discover and validate PUMA sample files without reading their contents."""

from __future__ import annotations

from pathlib import Path

from ...domain import PumaSample


def annotation_directory(root: Path, modality: str) -> Path:
    candidates = [
        root / f"geojson_{modality}",
        root / f"01_training_dataset_geojson_{modality}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No PUMA {modality} annotation directory under {root}")


def image_directory(root: Path) -> Path:
    candidates = [root / "images", root / "01_training_dataset_tif_ROIs"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No PUMA image directory under {root}")


def _annotation_path(directory: Path, sample_id: str, modality: str) -> Path:
    candidates = [
        directory / f"{sample_id}_{modality}.geojson",
        directory / f"{sample_id}.geojson",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {modality} annotation for sample {sample_id!r}")


def discover_puma_samples(root: str | Path) -> list[PumaSample]:
    root_path = Path(root)
    images = image_directory(root_path)
    tissue = annotation_directory(root_path, "tissue")
    nuclei = annotation_directory(root_path, "nuclei")
    image_paths = sorted(
        path for path in images.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {images}")
    samples = []
    for image_path in image_paths:
        sample_id = image_path.stem
        samples.append(
            PumaSample(
                sample_id=sample_id,
                image_path=image_path,
                tissue_annotation_path=_annotation_path(tissue, sample_id, "tissue"),
                nuclei_annotation_path=_annotation_path(nuclei, sample_id, "nuclei"),
            )
        )
    return samples
