"""Static test for training many multi-LoRA adapters together.

Creates 128 unique adapter configs from the gsm8k template, registers them
up front, then runs the standard train loop until --num-rollout.
"""

import asyncio
import logging
from pathlib import Path

import yaml

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.multi_lora_controller import create_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.adapter_config import ADAPTER_INACTIVE_STATES, ADAPTER_ROLLOUT_STATES
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


STATIC_ADAPTER_TEMPLATE = "gsm8k"
STATIC_ADAPTER_COUNT = 2
STATIC_ADAPTER_PREFIX = "gsm8k"


def prepare_static_adapter_dirs(multi_lora_dir: Path) -> list[Path]:
    """Create unique adapter configs that all train on the gsm8k dataset."""
    template_path = multi_lora_dir / STATIC_ADAPTER_TEMPLATE / "adapter.yaml"
    with open(template_path) as f:
        template = yaml.safe_load(f)

    adapter_dirs = []
    for i in range(STATIC_ADAPTER_COUNT):
        name = f"{STATIC_ADAPTER_PREFIX}_{i:02d}"
        adapter_dir = multi_lora_dir / "generated_static" / name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        config = dict(template)
        config["name"] = name

        with open(adapter_dir / "adapter.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        adapter_dirs.append(adapter_dir)

    return adapter_dirs


async def register_static_adapters(controller, multi_lora_dir: Path) -> None:
    """Register all static adapters before training starts."""
    adapter_dirs = prepare_static_adapter_dirs(multi_lora_dir)
    await asyncio.gather(
        *(controller.register_adapter.remote(str(adapter_dir)) for adapter_dir in adapter_dirs)
    )
    logger.info(f"registered {len(adapter_dirs)} static adapters from {STATIC_ADAPTER_TEMPLATE}")


async def run_trainer(args, controller, rollout_manager, actor_model, num_rollout_per_epoch) -> None:
    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    # sync starting weights to sglang
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    async def offload_train():
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def offload_rollout():
        # Offload if need to train or if need to update adapters
        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

    async def save(rollout_id):
        await actor_model.save_model(
            rollout_id,
            force_sync=rollout_id == args.num_rollout - 1,
        )
        if args.rollout_global_dataset:
            await rollout_manager.save.remote(rollout_id)

    rollout_id = args.start_rollout_id

    # Note: in colocated, rollout is inherently tied to train (1 rollout means 1 train) --
    # In async, we should have a run_rollout to gate the rollout.
    def should_run_train(adapter_configs):
        return any(config.state in ADAPTER_ROLLOUT_STATES for config in adapter_configs.values())

    def should_update_adapters(adapter_configs):
        return any(config.state in ADAPTER_INACTIVE_STATES for config in adapter_configs.values())

    # TODO: improve loop readability
    while True:
        adapter_configs = await controller.adapter_configs.remote()
        run_train = should_run_train(adapter_configs)
        update_adapters = should_update_adapters(adapter_configs)

        # Run training
        if run_train:
            rollout_data_ref = await rollout_manager.generate.remote(rollout_id)
            await offload_rollout()

            await actor_model.train(rollout_id, rollout_data_ref)
            await controller.report_training_completed.remote(rollout_id)

        # Load/unload adapteres
        if update_adapters:
            # Train already offloads the rollout
            if not run_train:
                await offload_rollout()
                await actor_model.onload()

            n_loaded = await actor_model.load_pending_adapters()
            n_unloaded = await actor_model.unload_drained_adapters(rollout_id)

        # Both cases need to push weights
        if run_train or update_adapters:
            # For run train, at the end, update rollout id and checkpoint if needed
            if run_train:
                if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                    await save(rollout_id)
                rollout_id += 1

            # Push the weights to sglang
            await offload_train()
            if args.offload_rollout:
                await rollout_manager.onload_weights.remote()
            await actor_model.update_weights()
            if args.offload_rollout:
                await rollout_manager.onload_kv.remote()
        else:
            print("Nothing to do: sleeping for 5s")
            await asyncio.sleep(5)

    await rollout_manager.dispose.remote()


async def main(args):
    configure_logger()
    if args.multi_lora_n_adapters < STATIC_ADAPTER_COUNT:
        raise ValueError(
            f"--multi-lora-n-adapters must be at least {STATIC_ADAPTER_COUNT} "
            f"for this static test, got {args.multi_lora_n_adapters}"
        )

    pgs = create_placement_groups(args)
    init_tracking(args)

    controller = create_multi_lora_controller(args.multi_lora_n_adapters, args.lora_rank)
    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"
    args.custom_generate_state_path = "miles.ray.multi_lora_controller.MultiLoRAGenerateState"

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    await register_static_adapters(controller, Path(args.multi_lora_dir))

    await run_trainer(args, controller, rollout_manager, actor_model, num_rollout_per_epoch)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
