"""Export PUMA polygons to the official CellViT++ SegmentationDataset contract."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ...domain import NucleusClass, NucleusInstance
from .dataset import read_native_image
from .discovery import discover_puma_samples
from .geojson import parse_nuclei_geojson
from .splits import create_split

__all__ = ["CELLVIT_PUMA_CLASS_ORDER", "CELLVIT_TYPE_ID", "export_cellvit_dataset", "rasterize_cellvit_instances"]

CELLVIT_PUMA_CLASS_ORDER = (
    NucleusClass.ENDOTHELIUM,
    NucleusClass.PLASMA_CELL,
    NucleusClass.STROMA,
    NucleusClass.TUMOR,
    NucleusClass.HISTIOCYTE,
    NucleusClass.APOPTOSIS,
    NucleusClass.EPITHELIUM,
    NucleusClass.MELANOPHAGE,
    NucleusClass.NEUTROPHIL,
    NucleusClass.LYMPHOCYTE,
)
CELLVIT_TYPE_ID = {label: index + 1 for index, label in enumerate(CELLVIT_PUMA_CLASS_ORDER)}


def rasterize_cellvit_instances(
    instances: list[NucleusInstance], image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint32 instance/type maps; zero is background in both maps."""
    height, width = image_size
    # OpenCV's polygon rasterizer supports signed int32 but not uint32.
    instance_map = np.zeros((height, width), dtype=np.int32)
    type_map = np.zeros((height, width), dtype=np.int32)
    for instance_id, instance in enumerate(instances, start=1):
        points = np.rint(instance.polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(instance_map, [points], instance_id)
        cv2.fillPoly(type_map, [points], CELLVIT_TYPE_ID[instance.label])
    return instance_map.astype(np.uint32), type_map.astype(np.uint32)


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    uint8 = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")


def _classifier_config(dataset_dir: Path, checkpoint: Path, output_dir: Path) -> str:
    labels = "\n".join(f"    '{i}': {label.value}" for i, label in enumerate(CELLVIT_PUMA_CLASS_ORDER))
    return f"""logging:
  mode: disabled
  project: prometheus-puma
  notes: CellViT-SAM-H-x40 PUMA Track-2
  log_comment: cellvit-sam-h-puma
  tags: [Classifier, PUMA]
  wandb_dir: {output_dir / "wandb"}
  log_dir: {output_dir}
  level: Info
random_seed: 42
gpu: 0
data:
  dataset: SegmentationDataset
  dataset_path: {dataset_dir}
  normalize_stains_train: false
  normalize_stains_val: false
  input_shape: [1024, 1024]
  num_classes: 10
  train_filelist: {dataset_dir / "splits" / "train.csv"}
  val_filelist: {dataset_dir / "splits" / "val.csv"}
  label_map:
{labels}
cellvit_path: {checkpoint}
training:
  cache_cell_dataset: true
  batch_size: 256
  epochs: 50
  drop_rate: 0.1
  optimizer: AdamW
  optimizer_hyperparameter:
    betas: [0.85, 0.9]
    lr: 0.00019395764571288664
    weight_decay: 0.0007665004192592943
  early_stopping_patience: 20
  mixed_precision: true
  eval_every: 1
  weighted_sampling: true
  scheduler:
    scheduler_type: exponential
  weight_list: [2, 5, 1.5, 1, 1.5, 5, 3, 3, 5, 1]
just_load_model: false
model:
  hidden_dim: 512
"""


def export_cellvit_dataset(
    data_root: str | Path,
    destination: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    """Write the CellViT++ ``SegmentationDataset`` layout, splits and classifier config.

    Returns a summary dict with the generated paths and sample counts.
    """
    destination = Path(destination).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    image_dir = destination / "train" / "images"
    label_dir = destination / "train" / "labels"
    split_dir = destination / "splits"
    for directory in (image_dir, label_dir, split_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples = discover_puma_samples(data_root)
    train_ids, validation_ids = create_split(samples, validation_fraction, seed)
    for sample in samples:
        image = read_native_image(sample.image_path)
        instances = parse_nuclei_geojson(sample.nuclei_annotation_path, strict=True)
        instance_map, type_map = rasterize_cellvit_instances(instances, image.shape[:2])
        _write_rgb_png(image_dir / f"{sample.sample_id}.png", image)
        # CellViT++ SegmentationDataset expects a pickled dict per label file, not an
        # array, so the object wrapper is part of their on-disk contract.
        np.save(
            label_dir / f"{sample.sample_id}.npy",
            np.asarray({"inst_map": instance_map, "type_map": type_map}, dtype=object),
            allow_pickle=True,
        )

    (split_dir / "train.csv").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (split_dir / "val.csv").write_text("\n".join(validation_ids) + "\n", encoding="utf-8")
    config_path = destination / "cellvit-puma-track2.yaml"
    config_path.write_text(_classifier_config(destination, checkpoint, output_dir), encoding="utf-8")
    return {
        "dataset_dir": str(destination),
        "config": str(config_path),
        "samples": len(samples),
        "train": len(train_ids),
        "validation": len(validation_ids),
    }
