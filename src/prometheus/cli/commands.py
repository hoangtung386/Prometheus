"""Command implementations. Argument parsing lives in :mod:`prometheus.cli.main`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..api import build_criterion, build_datamodule, build_model, build_trainer, load_config, load_predictor
from ..data.puma import (
    audit_puma_dataset,
    audit_resolution,
    audit_tissue_rasterization,
    export_cellvit_dataset,
    read_native_image,
)
from ..data.spatial import letterbox_image
from ..data.transforms import NormalizeMultitask, TransformSample
from ..domain import ImageMeta, Track
from ..engine import (
    assert_checkpoint_compatible,
    evaluate_multitask,
    load_engine_checkpoint,
    select_inference_state,
)
from ..inference import DIHEDRAL_VIEWS
from ..io import write_nuclei_json, write_tissue_tiff
from ..submission import validate_submission_outputs

__all__ = ["audit", "evaluate", "predict", "prepare_cellvit", "train"]


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def audit(args) -> int:
    """Run the dataset integrity, rasterization and resolution audits.

    Returns a non-zero exit code when an annotation fails to parse, so this is usable as a
    CI gate on a new dataset drop.
    """
    integrity = audit_puma_dataset(args.data_root)
    report: dict[str, object] = {"integrity": integrity}
    if not args.integrity_only:
        report["rasterization"] = audit_tissue_rasterization(args.data_root)
        report["resolution"] = audit_resolution(args.data_root)
    _print_json(report)
    return 1 if integrity["errors"] else 0


def train(args) -> int:
    """Train from a TOML config, optionally resuming from a checkpoint."""
    trainer = build_trainer(load_config(args.config))
    trainer.fit(resume_from=args.resume)
    return 0


def prepare_cellvit(args) -> int:
    """Export the dataset and classifier config for the CellViT++ nuclei path."""
    _print_json(
        export_cellvit_dataset(
            args.data_root,
            args.output,
            args.cellvit_checkpoint,
            args.run_dir,
            args.validation_fraction,
            args.seed,
        )
    )
    return 0


def _load_model(config, checkpoint_path: str, device: torch.device):
    model = build_model(config)
    checkpoint = load_engine_checkpoint(checkpoint_path, device)
    assert_checkpoint_compatible(checkpoint, config)
    model.load_state_dict(select_inference_state(checkpoint))
    return model.to(device)


def evaluate(args) -> int:
    """Evaluate a checkpoint on the validation split and print every metric."""
    config = load_config(args.config)
    _, validation_loader = build_datamodule(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_multitask(
        _load_model(config, args.checkpoint, device),
        validation_loader,
        build_criterion(config),
        device,
        config.model.nuclei_feature_stride,
        config.evaluation.nuclei_radius_px,
        config.postprocess.confidence_threshold,
        config.postprocess.max_detections,
        config.postprocess.local_max_kernel,
        tta_views=DIHEDRAL_VIEWS if args.tta else None,
    )
    _print_json({**result.metrics, "validation/loss": result.loss})
    return 0


def _prepare_image(path: Path, image_size: int) -> tuple[torch.Tensor, ImageMeta]:
    """Letterbox and normalize one image into a single-sample model input."""
    image, metadata = letterbox_image(read_native_image(path), (image_size, image_size), path.stem)
    valid_mask = np.zeros((image_size, image_size), dtype=bool)
    pad_x, pad_y = metadata.pad_xy
    height, width = metadata.resized_size
    valid_mask[pad_y : pad_y + height, pad_x : pad_x + width] = True
    normalized = NormalizeMultitask()(
        TransformSample(
            image=image.transpose(2, 0, 1),
            tissue_mask=np.empty((0, 0), dtype=np.uint8),
            centroids=np.empty((0, 2), dtype=np.float32),
            boxes=np.empty((0, 4), dtype=np.float32),
            valid_mask=valid_mask,
        )
    )
    return torch.from_numpy(normalized.image).float().unsqueeze(0), metadata


def predict(args) -> int:
    """Write the submission tissue TIFF and nuclei JSON for one image."""
    config = load_config(args.config)
    predictor = load_predictor(config, args.checkpoint)
    images, metadata = _prepare_image(Path(args.input), config.data.image_size)
    prediction = predictor.predict(images, [metadata])

    output_dir = Path(args.output)
    tissue_path = output_dir / "tissue.tif"
    nuclei_path = output_dir / "nuclei.json"
    write_tissue_tiff(prediction.tissue_masks[0], tissue_path)
    write_nuclei_json(prediction.nuclei[0], nuclei_path, Track(config.evaluation.track))
    validate_submission_outputs(tissue_path, nuclei_path)
    return 0
