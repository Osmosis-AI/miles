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

import logging
from typing import Mapping

import ray
import torch
import torch.distributed as dist

from miles.ray.multi_lora_controller import AdapterEntry, get_multi_lora_controller

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
    adapter_entries: Mapping[str, AdapterEntry],
):
    """Save per-adapter checkpoints to each adapter's directory."""
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
    from megatron.core import mpu

    from .hf_weight_iterator_bridge import HfWeightIteratorBridge

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    for adapter_name, entry in adapter_entries.items():
        ckpt_dir = entry.config.dir / "checkpoints" / f"step_{iteration}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        adapter_state = {}
        with expose_adapter_slot(model, entry.slot):
            iterator = HfWeightIteratorBridge(args=args, model=model, model_name=None, quantization_config=None, is_lora=True)
            for hf_named_tensors in iterator.get_hf_weight_chunks({}):
                for hf_name, weight in hf_named_tensors:
                    adapter_state[hf_name] = weight.cpu()

        native_path = ckpt_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        torch.save(adapter_state, native_path)
        logger.info(f"Saved adapter '{adapter_name}' checkpoint ({len(adapter_state)} tensors) to {native_path}")


def deregister_adapter(
    name: str,
    entry: AdapterEntry,
    rollout_id: int,
    args,
    model,
    optimizer,
    ipc_engine=None,
    ipc_gather_src=None,
):
    """Full cleanup for an exhausted adapter: save, unload, reset, deregister.

    Caller must pass the ``AdapterEntry`` from the snapshot they're processing —
    this function only writes to the controller (via ``deregister_run``) and
    does not re-snapshot.
    """
    from megatron.bridge.peft.multi_lora_layers import unregister_adapter

    from ..multi_lora import zero_optimizer_state_for_adapter

    save_multi_lora_checkpoints(args, model, rollout_id, {name: entry})
    logger.info(f"Saved final checkpoint for adapter '{name}'")

    if ipc_engine is not None and dist.get_rank() == ipc_gather_src:
        try:
            ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=name))
        except Exception:
            pass
    logger.info(f"Unloaded adapter '{name}' from SGLang")

    unregister_adapter(model, entry.slot)
    logger.info(f"Reset layer weights for adapter '{name}' (slot {entry.slot})")

    zero_optimizer_state_for_adapter(optimizer, model, entry.slot)
    optimizer.reload_model_params()

    ray.get(get_multi_lora_controller().deregister_run.remote(name))
    logger.info(f"Fully deregistered adapter '{name}'")
