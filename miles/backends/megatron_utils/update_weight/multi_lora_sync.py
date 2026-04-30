"""Multi-LoRA helpers: rank slicing, per-adapter checkpoint save, adapter cleanup.

The live per-step weight sync lives on ``UpdateWeightFromTensor`` itself
(``_send_multi_lora_params``); this module only holds the helpers that don't
naturally belong on that class.

TODO(perf): re-sync only adapters that were trained this step. Right now
``_send_multi_lora_params`` re-IPCs every registered adapter every call, even
when an adapter saw no data and its weights are unchanged. The hint already
exists in ``rollout_data["adapter_slots"]`` (set by the train actor); plumbing
it through cleanly without putting per-step state on the long-lived updater is
deferred — see chat 2026-04-28 for the design discussion.
"""

import json
import logging
import os
from collections.abc import Mapping

import ray
import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.ray.multi_lora_controller import get_multi_lora_controller
from miles.utils.adapter_config import AdapterConfig

logger = logging.getLogger(__name__)


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    """Slice a LoRA weight tensor from max_rank to adapter_rank for export.

    TODO: remove the zero-padding assertions once mixed-rank sync is validated.
    """
    if "lora_A" in hf_name and adapter_rank < tensor.shape[0]:
        remainder = tensor[adapter_rank:]
        assert remainder.abs().max() == 0, (
            f"lora_A padded dims are non-zero: {hf_name}, "
            f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
        )
        return tensor[:adapter_rank]
    if "lora_B" in hf_name and adapter_rank < tensor.shape[1]:
        remainder = tensor[:, adapter_rank:]
        assert remainder.abs().max() == 0, (
            f"lora_B padded dims are non-zero: {hf_name}, "
            f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
        )
        return tensor[:, :adapter_rank]
    return tensor


def save_multi_lora_checkpoints(
    args,
    model,
    iteration: int,
    adapter_configs: Mapping[str, AdapterConfig],
):
    """Save per-adapter checkpoints in two formats per adapter.

    Layout (per adapter)::

        {adapter.dir}/checkpoints/step_{iteration}/
        ├── adapter_megatron_tp{tp}_pp{pp}.pt   ← per-rank shard, fast resume
        ├── adapter_model.safetensors           ← gathered HF, inference / external
        └── adapter_config.json                 ← HF PEFT metadata (r, alpha, ...)

    The Megatron shard preserves the local TP/PP layout: each (tp, pp) tile
    writes its own file with Megatron-native parameter names, copied straight
    from the slot's ``ParallelLinearAdapter`` weights with no gather/scatter.
    Only one DP replica per tile writes (the others would write identical
    bytes). Resume is then a trivial ``param.data.copy_`` per tensor.

    The HF safetensors is TP-gathered by the bridge (collective across all
    ranks), then written by a single rank in standard PEFT layout so external
    tools (HuggingFace ``peft``, SGLang, vLLM) can consume it directly.

    Atomicity follows the single-LoRA pattern: ``dist.barrier()`` between
    sections so partial writes from a faster rank don't race with the next
    operation, plus ``os.sync()`` after the HF write to flush dirty pages
    before the function returns.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
    from megatron.core import mpu
    from safetensors.torch import save_file as save_safetensors

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf
    from miles.utils import megatron_bridge_utils

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    is_dp_rank_0 = get_parallel_state().intra_dp.rank == 0
    is_global_writer = is_dp_rank_0 and tp_rank == 0 and pp_rank == 0

    target_modules_hf = (
        convert_target_modules_to_hf(list(args.target_modules))
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Build the bridge once and reuse across every adapter — saves N-1
    # ``AutoBridge.from_hf_pretrained`` invocations for N adapters.
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    for adapter_name, config in adapter_configs.items():
        ckpt_dir = config.dir / "checkpoints" / f"step_{iteration}"
        if is_dp_rank_0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        with expose_adapter_slot(model, config.slot):
            # ---- (1) Megatron-native per-rank shard (fast resume) ----
            if is_dp_rank_0:
                shard: dict[str, torch.Tensor] = {
                    name: param.data.cpu()
                    for chunk in model
                    for name, param in chunk.named_parameters()
                    if ".adapter." in name
                }
                native_path = ckpt_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
                torch.save(shard, native_path)
                logger.info(
                    f"Saved adapter '{adapter_name}' Megatron shard "
                    f"({len(shard)} tensors) to {native_path}"
                )

            # ---- (2) HF PEFT format (TP-gathered, single file) ----
            # Bridge export is collective: every TP rank participates in the
            # all-gather. Only the global writer materialises the file.
            hf_state: dict[str, torch.Tensor] = {}
            with megatron_bridge_utils.patch_megatron_model(model):
                for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                    model, cpu=True, show_progress=False,
                ):
                    hf_state[hf_name] = weight.contiguous()

        if is_global_writer:
            save_safetensors(
                hf_state,
                str(ckpt_dir / "adapter_model.safetensors"),
                metadata={"format": "pt"},
            )
            adapter_config_json = {
                "peft_type": "LORA",
                "r": config.rank,
                "lora_alpha": config.alpha,
                "target_modules": target_modules_hf,
                "lora_dropout": getattr(args, "lora_dropout", 0.0),
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            with open(ckpt_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config_json, f, indent=2)
            os.sync()
            logger.info(
                f"Saved adapter '{adapter_name}' HF PEFT to {ckpt_dir} "
                f"({len(hf_state)} tensors)"
            )

        if dist.is_initialized():
            dist.barrier()


def deregister_adapter(
    name: str,
    config: AdapterConfig,
    rollout_id: int,
    args,
    model,
    optimizer,
    ipc_engine=None,
    ipc_gather_src=None,
):
    """Full cleanup for an exhausted adapter: save, unload, reset, deregister.

    Caller passes the ``AdapterConfig`` they're already iterating over — this
    function only writes to the controller (via ``deregister_run``) and does
    not re-fetch the adapter set.
    """
    from megatron.bridge.peft.multi_lora_layers import unregister_adapter

    from ..multi_lora import zero_optimizer_state_for_adapter

    save_multi_lora_checkpoints(args, model, rollout_id, {name: config})
    logger.info(f"Saved final checkpoint for adapter '{name}'")

    if ipc_engine is not None and dist.get_rank() == ipc_gather_src:
        try:
            ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=name))
        except Exception:
            pass
    logger.info(f"Unloaded adapter '{name}' from SGLang")

    unregister_adapter(model, config.slot)
    logger.info(f"Reset layer weights for adapter '{name}' (slot {config.slot})")

    zero_optimizer_state_for_adapter(optimizer, model, config.slot)
    optimizer.reload_model_params()

    ray.get(get_multi_lora_controller().deregister_run.remote(name))
    logger.info(f"Fully deregistered adapter '{name}'")
