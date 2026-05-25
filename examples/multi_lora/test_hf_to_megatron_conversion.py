"""Smoke test for HF -> Megatron-Bridge conversion with multi-LoRA FP8 args.

This intentionally stops after loading HuggingFace weights into the Megatron
model. It does not start rollout engines, register adapters, train, or push
weights back to SGLang.
"""

import asyncio
import logging
import os
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.backends.megatron_utils.initialize import init as init_megatron
from miles.backends.megatron_utils.initialize import is_megatron_main_rank
from miles.backends.megatron_utils.multi_lora import build_multi_lora_model, is_multi_lora_enabled
from miles.ray.placement_group import InfoActor, sort_key
from miles.ray.train_actor import TrainRayActor
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)


class HfToMegatronConversionActor(TrainRayActor):
    def convert(self, args):
        super().init(args, role="actor", with_ref=False)

        init_megatron(args)
        model = self._build_model(args)
        clear_memory()

        import miles_plugins.megatron_bridge  # noqa: F401
        from megatron.bridge import AutoBridge

        adapter_context = self._hide_adapters(model) if is_multi_lora_enabled(args) else nullcontext()
        with adapter_context, patch_megatron_model(model):
            bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
            bridge.load_hf_weights(model)

        dist.barrier()

        if is_megatron_main_rank():
            print_memory("after HF -> Megatron conversion")
            logger.info("HF -> Megatron conversion completed successfully")

        local_tensors = sum(1 for model_chunk in model for _ in model_chunk.named_parameters())
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_tensors": local_tensors,
        }

    def _build_model(self, args):
        if is_multi_lora_enabled(args):
            model, _multi_lora = build_multi_lora_model(args)
            return model

        from megatron.core.enums import ModelType
        from megatron.training.training import get_model

        from miles.backends.megatron_utils.model_provider import get_model_provider_func

        return get_model(get_model_provider_func(args, "actor"), ModelType.encoder_or_decoder)

    def _hide_adapters(self, model):
        from megatron.bridge.peft.multi_lora_layers import hide_adapters

        return hide_adapters(model)

    def sleep(self):
        raise NotImplementedError

    def wake_up(self):
        raise NotImplementedError

    def train(self, rollout_id, rollout_data_ref):
        raise NotImplementedError

    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError

    def connect_actor_critic(self, critic_group):
        raise NotImplementedError

    def _get_parallel_config(self):
        raise NotImplementedError


def _create_actor_placement_group(world_size: int):
    pg = placement_group([{"GPU": 1, "CPU": 1} for _ in range(world_size)], strategy="PACK")
    ray.get(pg.ready())

    info_actors = [
        InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
            )
        ).remote()
        for bundle_index in range(world_size)
    ]
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(index, node_ip, gpu_id) for index, (node_ip, gpu_id) in enumerate(gpu_ids)]
    reordered_bundle_indices = [info[0] for info in sorted(bundle_infos, key=sort_key)]
    return pg, reordered_bundle_indices


def _remote_actor_class(args):
    env_vars = {
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **args.train_env_vars,
    }
    return ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(HfToMegatronConversionActor)


async def main(args):
    configure_logger()
    ray.init(address="auto", ignore_reinit_error=True)

    world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    pg, bundle_indices = _create_actor_placement_group(world_size)
    ConversionActor = _remote_actor_class(args)

    actors = []
    master_addr, master_port = None, None
    for rank in range(world_size):
        actor = ConversionActor.options(
            num_cpus=1,
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_indices[rank],
            ),
        ).remote(world_size, rank, master_addr, master_port)
        if rank == 0:
            master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
        actors.append(actor)

    try:
        results = await asyncio.gather(*(actor.convert.remote(args) for actor in actors))
        if results:
            logger.info("Conversion results: %s", sorted(results, key=lambda item: item["rank"]))
    finally:
        for actor in actors:
            ray.kill(actor)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    asyncio.run(main(parse_args()))
