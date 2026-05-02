"""Multi-LoRA model setup and helpers for Megatron backend.

This module handles:
- Creating and configuring MultiLoRA instances
- Setting up the model with multi-adapter layers via megatron-bridge
- A wrapper around initialize_model_and_optimizer that also returns the MultiLoRA instance
- Optimizer state zeroing for adapter swap (future dynamic case)
"""

import dataclasses
import logging
from argparse import Namespace
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def is_multi_lora_enabled(args: Namespace) -> bool:
    return getattr(args, "multi_lora", False)


def create_multi_lora(args: Namespace):
    """Create a MultiLoRA instance from training args."""
    from megatron.bridge.peft.multi_lora import MultiLoRA

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_megatron

    lora_type_name = getattr(args, "lora_type", "lora").lower()
    if lora_type_name == "canonical_lora":
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA
        lora_cls = CanonicalLoRA
    else:
        from megatron.bridge.peft.lora import LoRA
        lora_cls = LoRA

    return MultiLoRA(
        target_modules=convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls),
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def build_multi_lora_model(args: Namespace):
    """Build Megatron model with MultiLoRA layers via megatron-bridge.

    Returns DDP-wrapped model chunks. Does NOT register adapters or load checkpoints —
    that happens after the optimizer is created.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig
    from transformers import AutoConfig

    from miles.backends.megatron_utils.bridge_lora_helpers import _make_value_model_hook

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    provider.finalize()

    multi_lora = create_multi_lora(args)

    def apply_hook(model_chunks):
        transformed = multi_lora(model_chunks, training=True)
        multi_lora.set_params_to_save(transformed)
        return transformed

    provider.register_pre_wrap_hook(apply_hook)

    is_value_model = (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        provider.register_pre_wrap_hook(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    ddp_config.finalize()

    if args.offload_train:
        from miles.backends.megatron_utils.lora_utils import patch_param_grad_buffer_for_colocate_mode_lora
        patch_param_grad_buffer_for_colocate_mode_lora()

    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)
    return model, multi_lora


def initialize_multi_lora_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
):
    """Drop-in alternative to initialize_model_and_optimizer for multi-LoRA.

    Builds model + optimizer, loads the base checkpoint, then installs every
    PENDING adapter the controller knows about *before* returning. This
    matters: the actor's ``weights_backuper.backup("actor")`` snapshots the
    model right after this returns, and the noop backuper asserts the hash
    is unchanged on every subsequent ``get("actor")`` — so any slot
    mutation that happens after the snapshot trips the assertion.

    Online additions (adapters registered after training starts) are
    installed by ``MegatronTrainRayActor.load_pending_adapters`` instead,
    which re-snapshots the backuper afterwards. Same return signature as
    ``initialize_model_and_optimizer``.
    """
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.ci_utils import check_model_hashes, check_peak_gpu_memory_after_load
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler
    from miles.utils.memory_utils import clear_memory

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from miles.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync

    model, multi_lora = build_multi_lora_model(args)
    model[0].role = role

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None
    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)

    # Hide adapter params so the bridge's conversion-task walk doesn't see them
    # while loading the base checkpoint.
    from megatron.bridge.peft.multi_lora_layers import hide_adapters

    clear_memory()
    with hide_adapters(model):
        iteration, _ = load_checkpoint(
            model, optimizer, opt_param_scheduler,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
    check_peak_gpu_memory_after_load(args)
    clear_memory()
    check_model_hashes(args, model, iteration)
    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    # Install every adapter the controller already knows about, before the
    # caller's backuper takes its snapshot. See the docstring for why.
    from .update_weight.multi_lora_sync import load_pending_adapters
    load_pending_adapters(args, model, optimizer)

    return model, optimizer, opt_param_scheduler, iteration


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Find the latest *cross-rank-complete* step checkpoint.

    Walks ``step_*`` directories from highest-numbered to lowest and returns
    the first one that contains the per-rank shard for *every* (tp, pp) tile,
    not just this rank's. This guarantees all ranks pick the same step on
    load — without it, asymmetric partial saves (rank 0 wrote, rank 1 didn't)
    would have ranks loading different versions of the same slot, silently
    corrupting the model. Each rank then returns its own ``tp{tp}_pp{pp}.pt``
    path from the chosen step.

    Atomic saves (``_tmp_step_N`` → ``step_N`` rename) make every fresh
    checkpoint complete by construction; this guard exists for legacy
    partial saves left on disk by the pre-atomic-save code.
    """
    if not ckpt_dir.exists():
        return None

    from megatron.core import mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    step_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )
    for step_dir in step_dirs:
        all_present = all(
            (step_dir / f"adapter_megatron_tp{tp}_pp{pp}.pt").exists()
            for tp in range(tp_size)
            for pp in range(pp_size)
        )
        if all_present:
            return step_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    return None


def zero_optimizer_state_for_adapter(optimizer, model, idx: int) -> None:
    """Zero Adam exp_avg/exp_avg_sq for a specific adapter slot's parameters."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, SimpleMultiLoRALinear, _iter_multi_lora_modules

    target_params = set()
    for module in _iter_multi_lora_modules(model):
        if isinstance(module, MultiLoRALinear):
            adapter = module.adapters[idx]
            for param in adapter.parameters():
                target_params.add(id(param))
        elif isinstance(module, SimpleMultiLoRALinear):
            adapter = module.adapters[idx]
            for param in adapter.parameters():
                target_params.add(id(param))

    for chained_optimizer in optimizer.chained_optimizers:
        for param, state in chained_optimizer.optimizer.state.items():
            if id(param) in target_params:
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
