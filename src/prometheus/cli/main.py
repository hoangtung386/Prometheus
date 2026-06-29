"""Single command-line entry point for Prometheus workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import load_experiment_config
from ..data import create_puma_dataloaders, test_transform, train_transform
from ..data.puma.audit import audit_puma_dataset
from ..data.puma.datasets import read_image
from ..domain import Track
from ..inference import PredictionPipeline
from ..io import write_nuclei_json, write_tissue_tiff
from ..losses import MulticlassCombinedLoss
from ..models import create_model
from ..submission import validate_submission_outputs
from ..training import Trainer, validate
from ..training.checkpointing import load_checkpoint


def _audit(args: argparse.Namespace) -> int:
    report = audit_puma_dataset(args.data_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


def _model_name(model_type: str) -> str:
    names = {"UNetTissue": "tissue_convnext_unet", "DualUNet": "legacy_dual_unet"}
    try:
        return names[model_type]
    except KeyError as error:
        raise ValueError(f"Unsupported model_type: {model_type}") from error


def _loaders(config):
    return create_puma_dataloaders(
        root=config.data.root,
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        test_split=config.data.validation_fraction,
        seed=config.data.split_seed,
        train_transforms=train_transform(),
        test_transforms=test_transform(),
        split_manifest_path=config.data.split_manifest,
        pin_memory=config.training.pin_memory,
    )


def _train(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    model = create_model(_model_name(config.training.model_type), config.model)
    train_loader, validation_loader = _loaders(config)
    Trainer(model, train_loader, config.training, test_loader=validation_loader).fit()
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    model = create_model(_model_name(config.training.model_type), config.model)
    model.load_state_dict(load_checkpoint(args.checkpoint)["model_state"])
    _, validation_loader = _loaders(config)
    criterion = {
        "tissue": MulticlassCombinedLoss(),
        "nuclei": MulticlassCombinedLoss(),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss, tissue_dice, nuclei_dice = validate(
        model,
        validation_loader,
        criterion,
        config.training,
        device,
    )
    print(json.dumps({"loss": loss, "tissue_dice": tissue_dice, "nuclei_dice": nuclei_dice}, indent=2))
    return 0


def _predict(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if config.training.model_type != "DualUNet":
        raise ValueError("Combined PUMA prediction currently requires DualUNet")
    model = create_model("legacy_dual_unet", config.model)
    model.load_state_dict(load_checkpoint(args.checkpoint)["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image, _ = read_image(Path(args.input), config.data.image_size)
    result = PredictionPipeline(model, device).predict(torch.from_numpy(image).unsqueeze(0))
    output = Path(args.output)
    nuclei_path = output / "nuclei.json"
    tissue_path = output / "tissue.tif"
    write_tissue_tiff(np.asarray(result.tissue_mask[0]), tissue_path)
    write_nuclei_json(result.nuclei[0], nuclei_path, Track(config.evaluation.track))
    validate_submission_outputs(tissue_path, nuclei_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prometheus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Validate a PUMA dataset")
    audit_parser.add_argument("--data-root", required=True)
    audit_parser.set_defaults(handler=_audit)
    train_parser = subparsers.add_parser("train", help="Train from a TOML config")
    train_parser.add_argument("--config", required=True)
    train_parser.set_defaults(handler=_train)
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.set_defaults(handler=_evaluate)
    predict_parser = subparsers.add_parser("predict", help="Run combined inference")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.set_defaults(handler=_predict)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))
