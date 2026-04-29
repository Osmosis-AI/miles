"""Multi-LoRA controller: singleton Ray actor managing adapter lifecycle.

The controller owns adapter lifecycle state — registration, slot allocation,
exhaustion tracking. It does **not** know about Megatron, SGLang, datasets,
or rewards. Consumers read the runtime state via a single ``adapter_configs()``
call that returns ``dict[str, AdapterConfig]`` (each ``AdapterConfig`` carries
its assigned ``slot`` and ``exhausted`` flag).

The driver creates the controller once via ``create_multi_lora_controller``
and any process can then look it up with ``get_multi_lora_controller`` (mirrors
the named-actor pattern used by ``miles.utils.prometheus_utils``).

When locked (during a training step), register/deregister calls are buffered
and applied on unlock, preventing race conditions.
"""

import dataclasses
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import ray

from miles.utils.adapter_config import AdapterConfig, parse_adapter_yaml

logger = logging.getLogger(__name__)

CONTROLLER_NAME = "miles_multi_lora_controller"


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
        self._adapter_configs: dict[str, AdapterConfig] = {}
        self.free_slots: set[int] = set(range(max_adapters))
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
        if config.name in self._adapter_configs:
            raise ValueError(f"Adapter '{config.name}' is already registered")
        if not self.free_slots:
            raise ValueError(f"No free adapter slots (max {self.max_adapters})")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self._adapter_configs[config.name] = dataclasses.replace(config, slot=slot)

        logger.info(f"Registered adapter '{config.name}' at slot {slot}")
        return {"name": config.name, "slot": slot}

    def deregister_run(self, name: str) -> int:
        """Deregister an adapter by name. Frees its slot for reuse.

        If locked, the call is buffered and applied on unlock.
        """
        if self.locked:
            self.pending.append(("deregister_run", (name,)))
            logger.info(f"Buffered deregister_run({name}) — controller is locked")
            return -1

        if name not in self._adapter_configs:
            raise KeyError(f"Adapter '{name}' is not registered")

        slot = self._adapter_configs[name].slot
        del self._adapter_configs[name]
        self.free_slots.add(slot)

        logger.info(f"Deregistered adapter '{name}' from slot {slot}")
        return slot

    def mark_exhausted(self, name: str) -> None:
        """Mark an adapter as exhausted (dataset finished). Called by the data source.

        The adapter remains active until a consumer calls ``deregister_run``.
        """
        if name in self._adapter_configs:
            self._adapter_configs[name] = dataclasses.replace(self._adapter_configs[name], exhausted=True)
            logger.info(f"Adapter '{name}' marked as exhausted")

    def adapter_configs(self) -> dict[str, AdapterConfig]:
        """Return a shallow copy of the current name → AdapterConfig mapping.

        Each ``AdapterConfig`` is frozen and carries its slot and exhausted flag.
        Consumers receive a fresh copy via Ray pickling, so mutations are safe.
        """
        return dict(self._adapter_configs)


@asynccontextmanager
async def controller_step_lock(controller):
    """Async context manager that locks the controller for the duration of a training step."""
    ray.get(controller.lock.remote())
    try:
        yield
    finally:
        ray.get(controller.unlock.remote())
