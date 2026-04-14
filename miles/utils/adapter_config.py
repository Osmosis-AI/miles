"""Adapter config parsing for multi-LoRA training.

Each adapter directory contains an adapter.yaml with per-adapter
identity (name, rank, alpha) and data path. Training-level config
(base model, target modules, max rank, LR) comes from CLI flags.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AdapterConfig:
    name: str
    rank: int
    alpha: int
    data: str
    dir: Path = field(default_factory=lambda: Path("."))


def parse_adapter_yaml(path: Path) -> AdapterConfig:
    """Parse a single adapter.yaml file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return AdapterConfig(
        name=raw["name"],
        rank=raw["rank"],
        alpha=raw["alpha"],
        data=raw["data"],
        dir=path.parent,
    )
