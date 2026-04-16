"""Multi-LoRA data source that wraps per-adapter data sources.

Implements the DataSource interface. Queries the MultiLoRAController for active
adapters, lazily creates/removes per-adapter RolloutDataSource instances, and
round-robins get_samples() across them. Each sample is stamped with adapter_name
and per-adapter metadata (rm_type).
"""

import copy
import logging
from argparse import Namespace
from pathlib import Path

import ray

from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class MultiLoRADataSource(DataSource):
    def __init__(self, args: Namespace):
        self.args = args
        self.controller = args.multi_lora_controller
        self.sources: dict[str, RolloutDataSource] = {}
        self.adapter_configs: dict[str, dict] = {}
        self._sync_from_controller()

    def _sync_from_controller(self):
        """Sync local data sources with the controller's active adapter set."""
        active = ray.get(self.controller.active_runs.remote())

        for name in list(self.sources.keys()):
            if name not in active:
                del self.sources[name]
                del self.adapter_configs[name]
                logger.info(f"Removed data source for adapter '{name}'")

        for name, cfg in active.items():
            if name not in self.sources:
                self.sources[name] = self._create_adapter_source(cfg)
                self.adapter_configs[name] = cfg
                logger.info(f"Created data source for adapter '{name}' from {cfg['data']}")

    def _create_adapter_source(self, cfg: dict) -> RolloutDataSource:
        """Create a RolloutDataSource for a single adapter's dataset."""
        adapter_args = copy.copy(self.args)
        adapter_args.prompt_data = cfg["data"]
        adapter_args.input_key = cfg.get("input_key", self.args.input_key)
        adapter_args.label_key = cfg.get("label_key", self.args.label_key)
        return RolloutDataSource(adapter_args)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Round-robin samples across active adapters, stamping each with adapter_name."""
        self._sync_from_controller()

        if not self.sources:
            return []

        adapter_names = list(self.sources.keys())
        per_adapter = num_samples // len(adapter_names)
        remainder = num_samples % len(adapter_names)

        all_samples = []
        for i, name in enumerate(adapter_names):
            count = per_adapter + (1 if i < remainder else 0)
            if count == 0:
                continue
            cfg = self.adapter_configs[name]
            adapter_samples = self.sources[name].get_samples(count)
            for group in adapter_samples:
                for sample in group:
                    sample.adapter_name = name
                    if cfg.get("rm_type"):
                        if sample.metadata is None:
                            sample.metadata = {}
                        sample.metadata["rm_type"] = cfg["rm_type"]
            all_samples.extend(adapter_samples)

        return all_samples

    def add_samples(self, samples: list[list[Sample]]):
        for group in samples:
            name = group[0].adapter_name if group else None
            if name and name in self.sources:
                self.sources[name].add_samples([group])

    def save(self, rollout_id):
        for source in self.sources.values():
            source.save(rollout_id)

    def load(self, rollout_id=None):
        for source in self.sources.values():
            source.load(rollout_id)
