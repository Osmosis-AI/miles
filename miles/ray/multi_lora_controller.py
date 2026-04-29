"""Multi-LoRA controller: singleton Ray actor managing adapter lifecycle.

The controller owns adapter lifecycle state — registration, slot allocation,
exhaustion tracking. It does **not** know about Megatron, SGLang, datasets,
or rewards. Consumers read the runtime state via a single ``snapshot()`` call
that returns an ``AdapterSnapshot``.

The driver creates the controller once via ``create_multi_lora_controller``
and any process can then look it up with ``get_multi_lora_controller`` (mirrors
the named-actor pattern used by ``miles.utils.prometheus_utils``).

When locked (during a training step), register/deregister calls are buffered
and applied on unlock, preventing race conditions.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import ray

from miles.utils.adapter_config import AdapterConfig, parse_adapter_yaml
from miles.utils.types import AdapterRef, RewardSpec

logger = logging.getLogger(__name__)

CONTROLLER_NAME = "miles_multi_lora_controller"


@dataclass(frozen=True)
class AdapterEntry:
    """An adapter as the controller sees it: parsed config + runtime slot."""

    config: AdapterConfig
    slot: int


@dataclass(frozen=True)
class AdapterSnapshot:
    """Frozen view of the controller's runtime state at a point in time.

    Returned by ``MultiLoRAController.snapshot()``. Cheap to pickle (members
    are themselves frozen dataclasses / dicts of frozen dataclasses).
    """

    entries: Mapping[str, AdapterEntry]
    exhausted: frozenset[str]

    def names(self) -> list[str]:
        return list(self.entries)

    def slot(self, name: str) -> int:
        return self.entries[name].slot

    def ref(self, name: str) -> AdapterRef:
        """Build the per-sample handle for ``name`` (identity + slot)."""
        return AdapterRef(name=name, slot=self.entries[name].slot)

    def reward_spec(self, name: str) -> RewardSpec:
        """Build the per-sample reward dispatch handle for ``name``."""
        cfg = self.entries[name].config
        return RewardSpec(rm_type=cfg.rm_type, custom_rm_path=cfg.custom_rm_path)


def create_multi_lora_controller(max_adapters: int, max_rank: int):
    """Create the named singleton controller. Call once from the driver."""
    return MultiLoRAController.options(name=CONTROLLER_NAME).remote(max_adapters, max_rank)


def get_multi_lora_controller():
    """Return the named controller handle. Call from anywhere after creation."""
    return ray.get_actor(CONTROLLER_NAME)


@ray.remote(num_cpus=0)
class MultiLoRAController:
    def __init__(self, max_adapters: int, max_rank: int):
        self.max_adapters = max_adapters
        self.max_rank = max_rank
        self.configs: dict[str, AdapterConfig] = {}
        self.slot_map: dict[str, int] = {}
        self.free_slots: set[int] = set(range(max_adapters))
        self.exhausted: set[str] = set()
        self.locked = False
        self.pending: list[tuple[str, tuple]] = []  # buffered (method, args) while locked

    def lock(self):
        """Lock the adapter set. Register/deregister calls are buffered until unlock."""
        self.locked = True

    def unlock(self):
        """Unlock and apply all buffered register/deregister calls."""
        self.locked = False
        results = []
        for method_name, args in self.pending:
            results.append(getattr(self, method_name)(*args))
        self.pending.clear()
        return results

    def register_run(self, adapter_dir: str) -> dict:
        """Register an adapter from its directory path.

        If locked, the call is buffered and applied on unlock.
        """
        if self.locked:
            self.pending.append(("register_run", (adapter_dir,)))
            logger.info(f"Buffered register_run({adapter_dir}) — controller is locked")
            return {"buffered": True}

        config = parse_adapter_yaml(Path(adapter_dir) / "adapter.yaml")

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
        """Deregister an adapter by name. Also clears it from the exhausted set.

        If locked, the call is buffered and applied on unlock.
        """
        if self.locked:
            self.pending.append(("deregister_run", (name,)))
            logger.info(f"Buffered deregister_run({name}) — controller is locked")
            return -1

        if name not in self.configs:
            raise KeyError(f"Adapter '{name}' is not registered")

        slot = self.slot_map.pop(name)
        del self.configs[name]
        self.free_slots.add(slot)
        self.exhausted.discard(name)

        logger.info(f"Deregistered adapter '{name}' from slot {slot}")
        return slot

    def mark_exhausted(self, name: str) -> None:
        """Mark an adapter as exhausted (dataset finished). Called by the data source.

        The adapter remains active until a consumer calls ``deregister_run``.
        """
        if name in self.configs:
            self.exhausted.add(name)
            logger.info(f"Adapter '{name}' marked as exhausted")

    def snapshot(self) -> AdapterSnapshot:
        """Return a frozen view of the current adapter set and exhaustion state."""
        entries = {
            name: AdapterEntry(config=self.configs[name], slot=self.slot_map[name])
            for name in self.configs
        }
        return AdapterSnapshot(entries=entries, exhausted=frozenset(self.exhausted))


@asynccontextmanager
async def controller_step_lock(controller):
    """Async context manager that locks the controller for the duration of a training step."""
    ray.get(controller.lock.remote())
    try:
        yield
    finally:
        ray.get(controller.unlock.remote())
