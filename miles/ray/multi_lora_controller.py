"""Multi-LoRA controller: singleton Ray actor owning the adapter registry
and lifecycle state machine. Knows nothing about Megatron / SGLang / datasets.

State machine: PENDING -> ACTIVE -> DRAINING -> DRAINED -> REMOVED.
The driver triggers register/deregister; the trainer triggers mark_active
(after install) and mark_removed (after cleanup). The trainer reports rollout
progress via report_generation_started / report_training_completed; those
advance the watermarks that move adapters DRAINING -> DRAINED.
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
        self.configs: dict[str, AdapterConfig] = {}
        self.free_slots: set[int] = set(range(max_adapters))
        self.last_started_rollout_id: int = -1
        self.last_trained_rollout_id: int = -1
        self.drain_until_rollout_id: dict[str, int] = {}

    def register_adapter(self, adapter_dir: str) -> dict:
        """Assign a slot and mark PENDING. Fails fast if no slots are free.

        The trainer picks up PENDING adapters in its next install pass and
        transitions them to ACTIVE via :meth:`mark_active`.
        """
        config = parse_adapter_yaml(Path(adapter_dir) / "adapter.yaml")
        assert config.rank <= self.max_rank, (
            f"Adapter '{config.name}' rank ({config.rank}) exceeds max rank ({self.max_rank})"
        )
        if config.name in self.configs:
            raise ValueError(f"Adapter '{config.name}' is already registered")
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")

        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.configs[config.name] = dataclasses.replace(
            config, slot=slot, state=AdapterState.PENDING
        )

        logger.info(f"Registered adapter '{config.name}' at slot {slot} (PENDING)")
        return {"name": config.name, "slot": slot}

    def mark_active(self, name: str) -> None:
        """Promote PENDING -> ACTIVE. Idempotent (no-op if already ACTIVE
        or no longer registered)."""
        if name not in self.configs:
            return
        cur = self.configs[name]
        if cur.state == AdapterState.ACTIVE:
            return
        if cur.state != AdapterState.PENDING:
            logger.warning(
                f"mark_active called on '{name}' in state {cur.state.name}; ignoring"
            )
            return
        self.configs[name] = dataclasses.replace(cur, state=AdapterState.ACTIVE)
        logger.info(f"Adapter '{name}' ACTIVE")

    def deregister_adapter(self, name: str) -> None:
        """Transition ACTIVE -> DRAINING (or straight to DRAINED) and snap the
        drain target.

        The drain target is the most recently started rollout_id, i.e. the
        last cycle that may have included this adapter. The adapter promotes
        to DRAINED once :meth:`report_training_completed` reports that cycle
        has finished training. If the trainer is already idle past that
        target (no in-flight rollout for this adapter), we skip DRAINING
        entirely so wait/idle phases don't get stuck holding a half-released
        slot.
        """
        if name not in self.configs:
            raise KeyError(f"Adapter '{name}' is not registered")
        cur = self.configs[name]
        if cur.state != AdapterState.ACTIVE:
            logger.info(f"Adapter '{name}' already in {cur.state.name}; ignoring deregister")
            return
        target = self.last_started_rollout_id
        self.drain_until_rollout_id[name] = target
        if self.last_trained_rollout_id >= target:
            self.configs[name] = dataclasses.replace(cur, state=AdapterState.DRAINED)
            logger.info(f"Adapter '{name}' DRAINED (immediate, drain_until_rollout_id={target})")
        else:
            self.configs[name] = dataclasses.replace(cur, state=AdapterState.DRAINING)
            logger.info(f"Adapter '{name}' DRAINING (drain_until_rollout_id={target})")

    def report_generation_started(self, rollout_id: int) -> None:
        if rollout_id > self.last_started_rollout_id:
            self.last_started_rollout_id = rollout_id

    def report_training_completed(self, rollout_id: int) -> None:
        """Bump last_trained_rollout_id; promote DRAINING adapters whose
        drain target is now met."""
        if rollout_id > self.last_trained_rollout_id:
            self.last_trained_rollout_id = rollout_id
        for name, target in list(self.drain_until_rollout_id.items()):
            if name not in self.configs:
                continue
            cur = self.configs[name]
            if cur.state != AdapterState.DRAINING:
                continue
            if self.last_trained_rollout_id >= target:
                self.configs[name] = dataclasses.replace(cur, state=AdapterState.DRAINED)
                logger.info(f"Adapter '{name}' DRAINED")

    def mark_removed(self, name: str) -> int:
        """Finalize removal: drop from registry and free the slot. Called by
        the orchestration layer once cross-system cleanup is done. Idempotent
        (returns ``-1`` if already removed) so it can fire from every train rank."""
        if name not in self.configs:
            return -1
        slot = self.configs[name].slot
        del self.configs[name]
        self.drain_until_rollout_id.pop(name, None)
        self.free_slots.add(slot)
        logger.info(f"Removed adapter '{name}' (slot {slot} freed)")
        return slot

    def adapter_configs(self) -> dict[str, AdapterConfig]:
        return dict(self.configs)

    def get_last_started_rollout_id(self) -> int:
        return self.last_started_rollout_id
