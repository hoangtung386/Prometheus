"""Small explicit model registry used by CLI composition code."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from ..config import ModelConfig

ModelFactory = Callable[[ModelConfig], nn.Module]
_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Model already registered: {name}")
    _REGISTRY[name] = factory


def create_model(name: str, config: ModelConfig) -> nn.Module:
    try:
        factory = _REGISTRY[name]
    except KeyError as error:
        choices = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model {name!r}; available: {choices}") from error
    config.validate()
    return factory(config)


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
