"""Multi-LoRA training that exercises online add/remove via a fixed schedule.

Drives a hard-coded sequence of register/deregister events to verify the
dynamic adapter lifecycle end-to-end:

  Phase 1: wait N iters — no adapters
  Phase 2: register dapo_math, run N productive cycles
  Phase 3: register gsm8k, run N productive cycles (both active)
  Phase 4: deregister dapo_math, run N productive cycles (gsm8k only)
  Phase 5: deregister gsm8k, wait N iters — no adapters
  Phase 6: register gsm8k (re-loads from checkpoint), run N productive cycles
  Phase 7: register dapo_math (re-loads from checkpoint), run remaining cycles

Adapters live on disk under ``--multi-lora-dir/<name>/adapter.yaml``.
This script does NOT register anything at startup — every register/
deregister fires from the schedule below.
"""

import asyncio
import logging
from dataclasses import dataclass, field
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
class Phase:
    """One step in the dynamic schedule.

    ``register`` / ``deregister`` fire once at the start of the phase.
    ``duration`` is the number of loop iterations to spend in this phase
    (counts both wait-loops and productive cycles); ``-1`` means "until
    the trainer reaches ``args.num_rollout``".
    """

    name: str
    register: tuple[str, ...] = field(default_factory=tuple)
    deregister: tuple[str, ...] = field(default_factory=tuple)
    duration: int = 3


PHASES: tuple[Phase, ...] = (
    Phase("wait_no_adapters", duration=3),
    Phase("load_dapo", register=("dapo_math",), duration=3),
    Phase("load_gsm8k", register=("gsm8k",), duration=3),
    Phase("unload_dapo", deregister=("dapo_math",), duration=3),
    Phase("unload_gsm8k_then_wait", deregister=("gsm8k",), duration=3),
    Phase("reload_gsm8k", register=("gsm8k",), duration=3),
    Phase("reload_dapo_until_end", register=("dapo_math",), duration=-1),
)


def _log_state(prefix: str, configs: dict) -> None:
    if not configs:
        logger.info(f"{prefix} no adapters")
        return
    parts = [f"{n}={c.state.name}(slot={c.slot})" for n, c in configs.items()]
    logger.info(f"{prefix} {', '.join(parts)}")


async def _drain_drained(args, controller, rollout_manager, actor_model, rollout_id: int) -> bool:
    """If any adapter is DRAINED, push to SGLang (which unloads it) and free
    the model-side slot so the same name can be re-registered later. Returns
    True if any cleanup happened.
    """
    configs = await controller.adapter_configs.remote()
    if not any(c.state == AdapterState.DRAINED for c in configs.values()):
        return False
    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()
    await actor_model.update_weights()
    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()
    await actor_model.unload_drained_adapters(rollout_id)
    return True


async def train(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # No startup registration — the schedule below drives every register
    # and deregister. The controller still needs to exist before any
    # consumer (rollout manager, train workers) tries to look it up.
    controller = create_multi_lora_controller(args.multi_lora_n_adapters, args.lora_rank)

    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, _critic_model = await create_training_models(args, pgs, rollout_manager)

    # Push base weights once so SGLang has something coherent to generate
    # with even before any adapter shows up.
    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()
    await actor_model.update_weights()
    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    rollout_id = args.start_rollout_id
    phase_idx = 0
    iters_in_phase = 0
    phase_started = False

    while rollout_id < args.num_rollout and phase_idx < len(PHASES):
        phase = PHASES[phase_idx]

        if not phase_started:
            logger.info(
                f"[scripted] >>> phase {phase_idx + 1}/{len(PHASES)} "
                f"'{phase.name}' (rollout_id={rollout_id})"
            )
            for name in phase.register:
                adapter_dir = Path(args.multi_lora_dir) / name
                ray.get(controller.register_adapter.remote(str(adapter_dir)))
                logger.info(f"[scripted] registered '{name}' from {adapter_dir}")
            for name in phase.deregister:
                ray.get(controller.deregister_adapter.remote(name))
                logger.info(f"[scripted] deregistered '{name}'")
            phase_started = True

        # Online additions: install model-side, then re-sync to SGLang so
        # this cycle's generate can reach the new adapter.
        n_installed = await actor_model.load_pending_adapters()
        if n_installed > 0:
            if args.offload_rollout:
                await rollout_manager.onload_weights.remote()
            await actor_model.update_weights()
            if args.offload_rollout:
                await rollout_manager.onload_kv.remote()

        configs = await controller.adapter_configs.remote()
        _log_state(f"[scripted] phase '{phase.name}' iter {iters_in_phase}:", configs)

        if AdapterState.ACTIVE not in {c.state for c in configs.values()}:
            # No active adapter — drain any leftovers from the previous
            # phase, then sleep one idle tick.
            if await _drain_drained(args, controller, rollout_manager, actor_model, rollout_id):
                logger.info(f"[scripted] drained DRAINED adapters during idle phase '{phase.name}'")
            await asyncio.sleep(args.multi_lora_idle_poll_s)
            iters_in_phase += 1
            if phase.duration != -1 and iters_in_phase >= phase.duration:
                phase_idx += 1
                iters_in_phase = 0
                phase_started = False
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

        # Push ACTIVE adapters and unload DRAINED ones from SGLang inside
        # update_weights' pause bracket; then free model-side slots.
        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
        await actor_model.update_weights()
        if args.offload_rollout:
            await rollout_manager.onload_kv.remote()

        await actor_model.unload_drained_adapters(rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

        rollout_id += 1
        iters_in_phase += 1

        if phase.duration != -1 and iters_in_phase >= phase.duration:
            phase_idx += 1
            iters_in_phase = 0
            phase_started = False

    logger.info(
        f"[scripted] all done at rollout_id={rollout_id} "
        f"(phase_idx={phase_idx}/{len(PHASES)})"
    )
    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train(args))
