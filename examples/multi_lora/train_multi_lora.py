"""Example multi-LoRA training script.

Trains two LoRA adapters (gsm8k + dapo_math) simultaneously on the same base model.
Each adapter has its own dataset, reward function, and checkpoint directory.

Usage:
    ray start --head --num-gpus 8
    ray job submit -- python examples/multi_lora/train_multi_lora.py \
        --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate \
        --hf-checkpoint Qwen/Qwen2.5-0.5B-Instruct \
        --lora-rank 32 --target-modules all-linear \
        --multi-lora-dir examples/multi_lora/adapters \
        --multi-lora-n-adapters 4 \
        --rollout-batch-size 32 --global-batch-size 256 \
        --num-rollout 100
"""

import asyncio
from pathlib import Path

import ray

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.multi_lora_controller import create_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


async def train(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # Create the named multi-LoRA controller and register adapters before
    # any consumer (rollout manager, train workers) tries to look it up.
    controller = create_multi_lora_controller(args.multi_lora_n_adapters, args.lora_rank)
    for adapter_dir in sorted(Path(args.multi_lora_dir).iterdir()):
        if (adapter_dir / "adapter.yaml").exists():
            ray.get(controller.register_adapter.remote(str(adapter_dir)))

    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    await actor_model.update_weights()

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        await controller.apply_pending_lifecycle.remote(rollout_id)
        await controller.report_generate_started.remote(rollout_id)

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

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train(args))
