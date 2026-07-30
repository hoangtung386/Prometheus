"""The CLI surface is a contract for batch pipelines, so its shape is pinned here."""

from __future__ import annotations

import pytest

from prometheus.cli import build_parser

EXPECTED_COMMANDS = {"audit", "prepare-cellvit", "train", "evaluate", "predict"}


def _subcommands(parser) -> set[str]:
    actions = [action for action in parser._actions if action.dest == "command"]
    return set(actions[0].choices)


def test_every_command_is_registered() -> None:
    assert _subcommands(build_parser()) == EXPECTED_COMMANDS


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["audit", "--data-root", "/data"], {"data_root": "/data", "integrity_only": False}),
        (["audit", "--data-root", "/data", "--integrity-only"], {"integrity_only": True}),
        (["train", "--config", "c.toml"], {"config": "c.toml", "resume": None}),
        (["evaluate", "--config", "c.toml", "--checkpoint", "b.ckpt"], {"tta": False}),
        (["evaluate", "--config", "c.toml", "--checkpoint", "b.ckpt", "--tta"], {"tta": True}),
    ],
)
def test_arguments_parse_to_the_expected_namespace(argv: list[str], expected: dict[str, object]) -> None:
    args = build_parser().parse_args(argv)

    for key, value in expected.items():
        assert getattr(args, key) == value
    assert callable(args.handler)


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_required_options_are_enforced() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train"])
