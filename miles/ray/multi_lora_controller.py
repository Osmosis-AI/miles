"""Multi-LoRA controller: singleton Ray actor managing adapter lifecycle.

The controller is the single source of truth for which adapters are active.
Training workers, the RolloutManager, and SGLang engines query it.

Adapters are registered explicitly via register_run(path) — no automatic
directory scanning.
"""

import logging

import ray

from miles.utils.adapter_config import parse_adapter_yaml

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0)
class MultiLoRAController:
    def __init__(self, max_adapters: int, max_rank: int):
        self.max_adapters = max_adapters
        self.max_rank = max_rank
        self.configs = {}
        self.slot_map = {}
        self.free_slots = set(range(max_adapters))

    def register_run(self, adapter_dir: str) -> dict:
        """Register an adapter from its directory path.

        The directory must contain an adapter.yaml file.
        Returns the config + slot assignment.
        """
        from pathlib import Path

        yaml_path = Path(adapter_dir) / "adapter.yaml"
        config = parse_adapter_yaml(yaml_path)

        assert config.rank <= self.max_rank, (
            f"Adapter '{config.name}' rank ({config.rank}) exceeds max rank ({self.max_rank})"
        )

        if config.name in self.configs:
            raise ValueError(f"Adapter '{config.name}' is already registered")
        if not self.free_slots:
            raise ValueError(f"No free adapter slots (max {self.max_adapters})")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.configs[config.name] = config
        self.slot_map[config.name] = slot

        logger.info(f"Registered adapter '{config.name}' at slot {slot}")
        return {"name": config.name, "slot": slot}

    def deregister_run(self, name: str) -> int:
        """Deregister an adapter by name. Returns the freed slot."""
        if name not in self.configs:
            raise KeyError(f"Adapter '{name}' is not registered")

        slot = self.slot_map.pop(name)
        del self.configs[name]
        self.free_slots.add(slot)

        logger.info(f"Deregistered adapter '{name}' from slot {slot}")
        return slot

    def active_runs(self) -> dict[str, dict]:
        """Return current adapter configs and slot assignments."""
        return {
            name: {
                "slot": self.slot_map[name],
                "rank": self.configs[name].rank,
                "alpha": self.configs[name].alpha,
                "data": self.configs[name].data,
                "dir": str(self.configs[name].dir),
            }
            for name in self.configs
        }
