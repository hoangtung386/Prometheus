"""Command-line composition root for the refactored Prometheus pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..api import build_datamodule, build_model, build_trainer, load_config, load_predictor
from ..data.puma.audit import audit_puma_dataset
from ..data.puma.multitask_dataset import read_native_image
from ..data.spatial import letterbox_image
from ..data.transforms import NormalizeMultitask, TransformSample
from ..domain import Track
from ..engine import assert_checkpoint_compatible, evaluate_multitask, load_engine_checkpoint
from ..io import write_nuclei_json, write_tissue_tiff
from ..losses import LossWeights, PrometheusMultitaskLoss
from ..submission import validate_submission_outputs


def _audit(args: argparse.Namespace) -> int:
    report = audit_puma_dataset(args.data_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


def _train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    trainer = build_trainer(config)
    trainer.fit(resume_from=args.resume)
    return 0


def _criterion(config) -> PrometheusMultitaskLoss:
    loss_weight_fields = {name for name in LossWeights.__dataclass_fields__}
    return PrometheusMultitaskLoss(
        config.model.num_nucleus_types,
        config.model.nuclei_feature_stride,
        LossWeights(**{name: value for name, value in config.loss.__dict__.items() if name in loss_weight_fields}),
    )


def _evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _, validation_loader = build_datamodule(config)
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_engine_checkpoint(args.checkpoint, device)
    assert_checkpoint_compatible(checkpoint, config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    result = evaluate_multitask(
        model,
        validation_loader,
        _criterion(config),
        device,
        config.model.nuclei_feature_stride,
        config.evaluation.nuclei_radius_px,
        config.postprocess.confidence_threshold,
        config.postprocess.max_detections,
        config.postprocess.local_max_kernel,
    )
    print(json.dumps(result.metrics | {"validation/loss": result.loss}, indent=2))
    return 0


def _prepare_image(path: Path, image_size: int):
    image, metadata = letterbox_image(read_native_image(path), (image_size, image_size), path.stem)
    chw = image.transpose(2, 0, 1)
    valid_mask = np.zeros((image_size, image_size), dtype=bool)
    pad_x, pad_y = metadata.pad_xy
    resized_height, resized_width = metadata.resized_size
    valid_mask[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = True
    transformed = NormalizeMultitask()(
        TransformSample(
            chw,
            np.empty((0, 0), dtype=np.uint8),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            valid_mask,
        )
    )
    return torch.from_numpy(transformed.image).float().unsqueeze(0), metadata


def _predict(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    predictor = load_predictor(config, args.checkpoint)
    image, metadata = _prepare_image(Path(args.input), config.data.image_size)
    prediction = predictor.predict(image, [metadata])
    output = Path(args.output)
    tissue_path = output / "tissue.tif"
    nuclei_path = output / "nuclei.json"
    write_tissue_tiff(prediction.tissue_masks[0], tissue_path)
    write_nuclei_json(prediction.nuclei[0], nuclei_path, Track(config.evaluation.track))
    validate_submission_outputs(tissue_path, nuclei_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prometheus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Validate labels and annotation integrity")
    audit_parser.add_argument("--data-root", required=True)
    audit_parser.set_defaults(handler=_audit)

    train_parser = subparsers.add_parser("train", help="Train PrometheusNet from a TOML config")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume")
    train_parser.set_defaults(handler=_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate an architecture-v2 checkpoint")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.set_defaults(handler=_evaluate)

    predict_parser = subparsers.add_parser("predict", help="Create source-space PUMA outputs")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.set_defaults(handler=_predict)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))
