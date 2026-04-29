"""Multi-LoRA data source that wraps per-adapter data sources.

Implements the DataSource interface. Queries the MultiLoRAController for the
current adapter snapshot, lazily creates/removes per-adapter ``RolloutDataSource``
instances, and round-robins ``get_samples()`` across them. Each emitted sample is
stamped with an ``AdapterRef`` (identity + slot) and a ``RewardSpec`` (per-adapter
reward dispatch); the same ref instances are shared across all samples of a
given adapter so they pickle-memoize on the wire.

When an adapter's dataset reaches its configured ``max_epochs``, the adapter is
marked exhausted on the controller; the train actor performs the actual cleanup.
"""

import copy
import logging
from argparse import Namespace

import ray

from miles.ray.multi_lora_controller import AdapterEntry, get_multi_lora_controller
from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class MultiLoRADataSource(DataSource):
    def __init__(self, args: Namespace):
        self.args = args
        self.controller = get_multi_lora_controller()
        self.sources: dict[str, RolloutDataSource] = {}
        self.entries: dict[str, AdapterEntry] = {}
        self.epoch_counts: dict[str, int] = {}
        self._reconcile(self._snapshot_entries())

    def _snapshot_entries(self) -> dict[str, AdapterEntry]:
        snapshot = ray.get(self.controller.snapshot.remote())
        return dict(snapshot.entries)

    def _reconcile(self, entries: dict[str, AdapterEntry]) -> None:
        """Add data sources for newly-registered adapters; drop ones no longer active."""
        for name in list(self.sources):
            if name not in entries:
                del self.sources[name]
                del self.entries[name]
                del self.epoch_counts[name]
                logger.info(f"Removed data source for adapter '{name}'")

        for name, entry in entries.items():
            if name not in self.sources:
                self.sources[name] = self._create_adapter_source(entry)
                self.entries[name] = entry
                self.epoch_counts[name] = 0
                logger.info(f"Created data source for adapter '{name}' from {entry.config.data}")

    def _create_adapter_source(self, entry: AdapterEntry) -> RolloutDataSource:
        cfg = entry.config
        adapter_args = copy.copy(self.args)
        adapter_args.prompt_data = cfg.data
        adapter_args.input_key = cfg.input_key or self.args.input_key
        adapter_args.label_key = cfg.label_key or self.args.label_key
        return RolloutDataSource(adapter_args)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        snapshot = ray.get(self.controller.snapshot.remote())
        self._reconcile(dict(snapshot.entries))

        if not self.sources:
            return []

        adapter_names = list(self.sources)
        per_adapter = num_samples // len(adapter_names)
        remainder = num_samples % len(adapter_names)

        # Build refs once per adapter so all samples of the same adapter share
        # the same instance (pickle memoizes by identity).
        refs = {name: snapshot.ref(name) for name in adapter_names}
        reward_specs = {name: snapshot.reward_spec(name) for name in adapter_names}

        all_samples: list[list[Sample]] = []
        exhausted: list[str] = []

        for i, name in enumerate(adapter_names):
            count = per_adapter + (1 if i < remainder else 0)
            if count == 0:
                continue

            source = self.sources[name]
            entry = self.entries[name]
            prev_epoch = source.epoch_id

            adapter_samples = source.get_samples(count)

            if source.epoch_id > prev_epoch:
                self.epoch_counts[name] = source.epoch_id
                max_epochs = entry.config.max_epochs
                if max_epochs is not None and source.epoch_id >= max_epochs:
                    logger.info(f"Adapter '{name}' reached max_epochs={max_epochs}, will deregister")
                    exhausted.append(name)

            ref = refs[name]
            reward_spec = reward_specs[name]
            for group in adapter_samples:
                for sample in group:
                    sample.adapter = ref
                    sample.reward_spec = reward_spec
            all_samples.extend(adapter_samples)

        for name in exhausted:
            ray.get(self.controller.mark_exhausted.remote(name))

        return all_samples

    def add_samples(self, samples: list[list[Sample]]):
        for group in samples:
            name = group[0].adapter.name if group and group[0].adapter else None
            if name and name in self.sources:
                self.sources[name].add_samples([group])

    def save(self, rollout_id):
        for source in self.sources.values():
            source.save(rollout_id)

    def load(self, rollout_id=None):
        for source in self.sources.values():
            source.load(rollout_id)
