"""Multi-LoRA data source that wraps per-adapter data sources.

Implements the DataSource interface. Queries the MultiLoRAController for active
adapters, lazily creates/removes per-adapter RolloutDataSource instances, and
round-robins get_samples() across them. Each sample is stamped with adapter_name.
"""

import copy
import itertools
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
        self.adapter_dirs: dict[str, str] = {}
        self._sync_from_controller()

    def _sync_from_controller(self):
        """Sync local data sources with the controller's active adapter set."""
        active = ray.get(self.controller.active_runs.remote())

        # Remove data sources for deregistered adapters
        for name in list(self.sources.keys()):
            if name not in active:
                del self.sources[name]
                del self.adapter_dirs[name]
                logger.info(f"Removed data source for adapter '{name}'")

        # Add data sources for newly registered adapters
        for name, cfg in active.items():
            if name not in self.sources:
                adapter_dir = Path(cfg["dir"])
                data_path = str(adapter_dir / "dataset.jsonl")
                self.sources[name] = self._create_adapter_source(data_path)
                self.adapter_dirs[name] = cfg["dir"]
                logger.info(f"Created data source for adapter '{name}' from {data_path}")

    def _create_adapter_source(self, data_path: str) -> RolloutDataSource:
        """Create a RolloutDataSource for a single adapter's dataset."""
        adapter_args = copy.copy(self.args)
        adapter_args.prompt_data = data_path
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
            adapter_samples = self.sources[name].get_samples(count)
            for group in adapter_samples:
                for sample in group:
                    sample.adapter_name = name
            all_samples.extend(adapter_samples)

        return all_samples

    def add_samples(self, samples: list[list[Sample]]):
        """Route samples back to the appropriate adapter's data source."""
        for group in samples:
            name = group[0].adapter_name if group else None
            if name and name in self.sources:
                self.sources[name].add_samples([group])

    def save(self, rollout_id):
        for name, source in self.sources.items():
            source.save(rollout_id)

    def load(self, rollout_id=None):
        for name, source in self.sources.items():
            source.load(rollout_id)
