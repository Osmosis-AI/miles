"""Adapter config parsing for multi-LoRA training.

Scans a directory for adapter.yaml files and parses per-adapter
identity (name, rank, alpha) and data path. Training-level config
(base model, target modules, max rank, LR) comes from CLI flags.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Parsed adapter configuration from adapter.yaml."""

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


def scan_adapter_dir(multi_lora_dir: str | Path) -> list[AdapterConfig]:
    """Scan a directory for adapter.yaml files and parse them.

    Returns a list of AdapterConfig sorted by name for deterministic slot ordering.
    """
    root = Path(multi_lora_dir)
    assert root.is_dir(), f"--multi-lora-dir {root} is not a directory"

    configs = []
    for adapter_yaml in sorted(root.glob("*/adapter.yaml")):
        configs.append(parse_adapter_yaml(adapter_yaml))

    assert len(configs) > 0, f"No adapter.yaml files found in {root}"
    return configs


def validate_adapter_configs(configs: list[AdapterConfig], max_rank: int) -> None:
    """Validate adapter configs against training-level settings."""
    for c in configs:
        assert c.rank <= max_rank, (
            f"Adapter '{c.name}' rank ({c.rank}) exceeds --lora-rank ({max_rank})"
        )
        assert c.rank > 0, f"Adapter '{c.name}' rank must be > 0"
        assert c.alpha > 0, f"Adapter '{c.name}' alpha must be > 0"


def load_multi_lora_configs(multi_lora_dir: str | Path, max_rank: int) -> list[AdapterConfig]:
    """Scan, parse, and validate all adapter configs in a directory."""
    configs = scan_adapter_dir(multi_lora_dir)
    validate_adapter_configs(configs, max_rank)
    logger.info(
        f"Found {len(configs)} adapters in {multi_lora_dir}: "
        f"{[c.name for c in configs]}"
    )
    return configs
