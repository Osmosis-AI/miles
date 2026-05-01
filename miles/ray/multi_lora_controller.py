"""Multi-LoRA controller: singleton Ray actor owning the adapter registry
and lifecycle state machine. Knows nothing about Megatron / SGLang / datasets.

State transitions are driven by the trainer via ``apply_pending_lifecycle``
(at the top of each rollout cycle) and the ``report_*`` methods.
"""

import dataclasses
import logging
from pathlib import Path

import ray

from miles.utils.adapter_config import AdapterConfig, AdapterState, parse_adapter_yaml

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
        self._max_initiated: int = -1
        self._max_trained: int = -1
        self._drain_target: dict[str, int] = {}
        self._pending: list[tuple[str, str]] = []  # (action, name)

    def register_adapter(self, adapter_dir: str) -> dict:
        """Assign a slot and mark ACTIVE. Fails fast if no slots are free."""
        config = parse_adapter_yaml(Path(adapter_dir) / "adapter.yaml")
        assert config.rank <= self.max_rank, (
            f"Adapter '{config.name}' rank ({config.rank}) exceeds max rank ({self.max_rank})"
        )
        if config.name in self._adapter_configs:
            raise ValueError(f"Adapter '{config.name}' is already registered")
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self._adapter_configs[config.name] = dataclasses.replace(
            config, slot=slot, state=AdapterState.ACTIVE
        )

        logger.info(f"Registered adapter '{config.name}' at slot {slot} (ACTIVE)")
        return {"name": config.name, "slot": slot}

    def deregister_adapter(self, name: str) -> None:
        """Buffer a drain request; applied at the next lifecycle gate."""
        if name not in self._adapter_configs:
            raise KeyError(f"Adapter '{name}' is not registered")
        cur_state = self._adapter_configs[name].state
        if cur_state != AdapterState.ACTIVE:
            logger.info(f"Adapter '{name}' already in {cur_state.name}; ignoring deregister")
            return
        self._pending.append(("deregister", name))
        logger.info(f"Buffered deregister for adapter '{name}'")

    def apply_pending_lifecycle(self, rollout_id: int) -> None:
        """Drain buffered requests. Must run *before* ``report_generate_started``
        so ``drain_target`` snaps to the last cycle that included the adapter."""
        for action, name in self._pending:
            if action == "deregister":
                if name not in self._adapter_configs:
                    continue
                cur = self._adapter_configs[name]
                if cur.state != AdapterState.ACTIVE:
                    continue
                self._adapter_configs[name] = dataclasses.replace(cur, state=AdapterState.DRAINING)
                self._drain_target[name] = self._max_initiated
                logger.info(f"Adapter '{name}' DRAINING (drain_target={self._max_initiated})")
        self._pending.clear()

    def report_generate_started(self, rollout_id: int) -> None:
        if rollout_id > self._max_initiated:
            self._max_initiated = rollout_id

    def report_train_completed(self, rollout_id: int) -> None:
        """Bump max_trained; promote DRAINING adapters whose target is met."""
        if rollout_id > self._max_trained:
            self._max_trained = rollout_id
        for name, target in list(self._drain_target.items()):
            if name not in self._adapter_configs:
                continue
            cur = self._adapter_configs[name]
            if cur.state != AdapterState.DRAINING:
                continue
            if self._max_trained >= target:
                self._adapter_configs[name] = dataclasses.replace(cur, state=AdapterState.DRAINED)
                logger.info(f"Adapter '{name}' DRAINED")

    def mark_removed(self, name: str) -> int:
        """Finalize removal: drop from registry and free the slot. Called by
        the orchestration layer once cross-system cleanup is done."""
        if name not in self._adapter_configs:
            raise KeyError(f"Adapter '{name}' is not registered")
        slot = self._adapter_configs[name].slot
        del self._adapter_configs[name]
        self._drain_target.pop(name, None)
        self.free_slots.add(slot)
        logger.info(f"Removed adapter '{name}' (slot {slot} freed)")
        return slot

    def adapter_configs(self) -> dict[str, AdapterConfig]:
        return dict(self._adapter_configs)
