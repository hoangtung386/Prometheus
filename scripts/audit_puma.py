#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from prometheus.data.puma_dataset import (
    NUCLEI_CLASS_TO_IDX,
    NUCLEI_CLASSES,
    TISSUE_CLASS_TO_IDX,
    TISSUE_CLASSES,
    PUMADataset,
    _parse_puma_class_name,
)


def _geojson_dir(root: Path, modality: str) -> Path:
    path = root / f"geojson_{modality}"
    if path.exists():
        return path
    return root / f"01_training_dataset_geojson_{modality}"


def _scan_labels(root: Path, modality: str) -> tuple[Counter[str], Counter[str], Counter[str]]:
    class_map = TISSUE_CLASS_TO_IDX if modality == "tissue" else NUCLEI_CLASS_TO_IDX
    raw_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    for path in sorted(_geojson_dir(root, modality).glob("*.geojson")):
        with open(path) as f:
            data = json.load(f)
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            raw = props.get("label")
            if raw is None:
                classification = props.get("classification", {})
                raw = classification.get("name", "background") if isinstance(classification, dict) else "background"
            raw = str(raw)
            mapped = _parse_puma_class_name(raw)
            raw_counts[raw] += 1
            if mapped in class_map:
                mapped_counts[mapped] += 1
            else:
                unknown_counts[raw] += 1
    return raw_counts, mapped_counts, unknown_counts


def _pixel_distribution(dataset: PUMADataset, modality: str, class_names: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for idx in range(len(dataset)):
        mask = dataset._load_mask(idx, modality)
        target = torch.from_numpy(mask.argmax(axis=0))
        values = torch.bincount(target.reshape(-1), minlength=len(class_names))
        for cls_idx, value in enumerate(values.tolist()):
            counts[cls_idx] += int(value)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PUMA labels and class balance")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()

    root = Path(args.data_root)
    dataset = PUMADataset(root=root, image_size=args.image_size, augment=False, cache_masks=True)
    print(f"Samples: {len(dataset)}")

    for modality, class_names in [("tissue", TISSUE_CLASSES), ("nuclei", NUCLEI_CLASSES)]:
        print(f"\n[{modality}] labels")
        raw_counts, mapped_counts, unknown_counts = _scan_labels(root, modality)
        print(f"  Raw labels: {dict(raw_counts)}")
        print(f"  Mapped labels: {dict(mapped_counts)}")
        if unknown_counts:
            print(f"  Unknown labels: {dict(unknown_counts)}")

        pixel_counts = _pixel_distribution(dataset, modality, class_names)
        total = sum(pixel_counts.values())
        print(f"  Pixel distribution ({total:,} px):")
        for cls_idx, name in enumerate(class_names):
            count = pixel_counts[cls_idx]
            ratio = count / max(total, 1) * 100
            print(f"    {name}: {ratio:.4f}% ({count:,})")


if __name__ == "__main__":
    main()
