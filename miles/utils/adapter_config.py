"""Adapter config parsing for multi-LoRA training.

Each adapter directory contains an adapter.yaml with per-adapter
identity (name, rank, alpha), data path, and dataset-specific keys.
Training-level config (base model, target modules, max rank, LR)
comes from CLI flags.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AdapterConfig:
    """Per-adapter config + controller-assigned runtime state.

    Yaml-derived fields (name, rank, alpha, data, etc.) come from ``parse_adapter_yaml``.
    Runtime fields (slot, exhausted) are set by the ``MultiLoRAController`` post-parse
    via ``dataclasses.replace``. Frozen so consumers can safely receive shared copies.
    """

    name: str
    rank: int
    alpha: int
    data: str
    dir: Path = field(default_factory=lambda: Path("."))
    input_key: str = "text"
    label_key: str | None = None
    rm_type: str | None = None
    custom_rm_path: str | None = None
    max_epochs: int | None = None
    slot: int = -1
    exhausted: bool = False


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
        input_key=raw.get("input_key", "text"),
        label_key=raw.get("label_key"),
        rm_type=raw.get("rm_type"),
        custom_rm_path=raw.get("custom_rm_path"),
        max_epochs=raw.get("max_epochs"),
    )
