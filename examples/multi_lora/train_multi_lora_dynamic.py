"""Dynamic test for the multi-LoRA online add/remove lifecycle.

Runs the standard train loop alongside a small scheduler task that fires
register/deregister events at predefined points. The trainer reacts via
its existing lifecycle hooks (``load_pending_adapters``, the idle gate,
``unload_drained_adapters``) — it has no knowledge of the schedule.

Schedule:
  1. idle 30s (no adapters)
  2. register dapo_math   -> wait 3 productive cycles
  3. register gsm8k       -> wait 3 productive cycles (both active)
  4. deregister dapo_math -> wait 3 productive cycles (gsm8k only)
  5. deregister gsm8k     -> idle 30s (no adapters)
  6. register gsm8k       -> wait 3 productive cycles
  7. register dapo_math   -> trainer runs to --num-rollout
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import ray

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.multi_lora_controller import create_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.adapter_config import AdapterState
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


@dataclass
class Step:
    name: str
    register: tuple[str, ...] = ()
    deregister: tuple[str, ...] = ()
    wait_cycles: int = 0
    wait_seconds: float = 0.0


SCHEDULE: tuple[Step, ...] = (
    Step("idle1",              wait_seconds=30.0),
    Step("load_dapo",          register=("dapo_math",), wait_cycles=3),
    Step("load_gsm8k",         register=("gsm8k",),     wait_cycles=3),
    Step("unload_dapo",        deregister=("dapo_math",), wait_cycles=3),
    Step("unload_gsm8k_idle",  deregister=("gsm8k",),   wait_seconds=30.0),
    Step("reload_gsm8k",       register=("gsm8k",),     wait_cycles=3),
    Step("reload_dapo_to_end", register=("dapo_math",)),
)


async def run_schedule(controller, multi_lora_dir: Path) -> None:
    """Drive register/deregister events. Talks only to the controller."""
    for step in SCHEDULE:
        logger.info(f"[schedule] >>> {step.name}")
        for name in step.register:
            ray.get(controller.register_adapter.remote(str(multi_lora_dir / name)))
            logger.info(f"[schedule] registered {name}")
        for name in step.deregister:
            ray.get(controller.deregister_adapter.remote(name))
            logger.info(f"[schedule] deregistered {name}")

        if step.wait_seconds > 0:
            await asyncio.sleep(step.wait_seconds)
        if step.wait_cycles > 0:
            start = await controller.get_last_started_rollout_id.remote()
            target = start + step.wait_cycles
            while True:
                cur = await controller.get_last_started_rollout_id.remote()
                if cur >= target:
                    break
                await asyncio.sleep(2.0)
        logger.info(f"[schedule] <<< {step.name} done")
    logger.info("[schedule] all steps done; trainer continues to --num-rollout")


async def run_trainer(args, controller, rollout_manager, actor_model, num_rollout_per_epoch) -> None:
    """Standard multi-LoRA train loop. Identical structure to
    train_multi_lora.py — the dynamic test doesn't change the trainer."""
    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()
    await actor_model.update_weights()
    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    rollout_id = args.start_rollout_id
    while rollout_id < args.num_rollout:
        n_installed = await actor_model.load_pending_adapters()
        if n_installed > 0:
            await actor_model.update_weights()

        configs = await controller.adapter_configs.remote()
        if AdapterState.ACTIVE not in {c.state for c in configs.values()}:
            if any(c.state == AdapterState.DRAINED for c in configs.values()):
                await actor_model.update_weights()
                await actor_model.unload_drained_adapters(rollout_id)
            await asyncio.sleep(args.multi_lora_idle_poll_s)
            continue

        await controller.report_generation_started.remote(rollout_id)

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        rollout_data_ref = await rollout_manager.generate.remote(rollout_id)

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

        await actor_model.train(rollout_id, rollout_data_ref)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await actor_model.save_model(rollout_id)

        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
        await actor_model.update_weights()
        if args.offload_rollout:
            await rollout_manager.onload_kv.remote()

        await actor_model.unload_drained_adapters(rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

        rollout_id += 1

    await rollout_manager.dispose.remote()


async def main(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # No startup registration — the schedule task drives all events.
    controller = create_multi_lora_controller(args.multi_lora_n_adapters, args.lora_rank)
    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, _critic_model = await create_training_models(args, pgs, rollout_manager)

    await asyncio.gather(
        run_trainer(args, controller, rollout_manager, actor_model, num_rollout_per_epoch),
        run_schedule(controller, Path(args.multi_lora_dir)),
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
