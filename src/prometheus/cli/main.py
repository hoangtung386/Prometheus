"""Argument parsing and process entry point for the ``prometheus`` command."""

from __future__ import annotations

import argparse

from . import commands

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the full ``prometheus`` parser. Exposed so tests can assert the interface."""
    parser = argparse.ArgumentParser(prog="prometheus", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit labels, tissue rasterization and resolution")
    audit.add_argument("--data-root", required=True)
    audit.add_argument(
        "--integrity-only",
        action="store_true",
        help="Only parse annotations; skip the rasterization and resolution audits",
    )
    audit.set_defaults(handler=commands.audit)

    cellvit = subparsers.add_parser(
        "prepare-cellvit", help="Export PUMA and generate a CellViT++ Track-2 classifier config"
    )
    cellvit.add_argument("--data-root", required=True)
    cellvit.add_argument("--output", required=True)
    cellvit.add_argument("--cellvit-checkpoint", required=True)
    cellvit.add_argument("--run-dir", required=True)
    cellvit.add_argument("--validation-fraction", type=float, default=0.2)
    cellvit.add_argument("--seed", type=int, default=42)
    cellvit.set_defaults(handler=commands.prepare_cellvit)

    train = subparsers.add_parser("train", help="Train PrometheusNet from a TOML config")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", help="Checkpoint to resume from; defaults to a fresh run")
    train.set_defaults(handler=commands.train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an architecture-v2 checkpoint")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--tta", action="store_true", help="Average over the eight dihedral views")
    evaluate.set_defaults(handler=commands.evaluate)

    predict = subparsers.add_parser("predict", help="Create source-space PUMA submission outputs")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.set_defaults(handler=commands.predict)
    return parser


def main() -> int:
    """Parse ``sys.argv`` and dispatch; returns the process exit code."""
    args = build_parser().parse_args()
    return int(args.handler(args))
