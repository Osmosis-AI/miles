"""Multi-LoRA model-side orchestration.

Two lifecycle hooks the train script calls each cycle, both *model-side only*:

- ``load_pending_adapters``  : PENDING -> ACTIVE (init slot + load ckpt)
- ``unload_drained_adapters``: DRAINED -> REMOVED (save + clear slot + zero opt)

Every SGLang interaction (push for ACTIVE, unload for DRAINED) lives inside
``UpdateWeightFromTensor._send_multi_lora_params`` so it free-rides on the
existing pause / flush / continue bracket of ``update_weights()`` — no new
bracket logic anywhere.

Plus checkpoint helpers (``save_multi_lora_checkpoints``, ``slice_lora_to_rank``).

TODO(perf): re-sync only adapters trained this step. ``_send_multi_lora_params``
currently re-IPCs every ACTIVE adapter every call, even when its weights are
unchanged. The hint already exists in ``rollout_data["adapter_slots"]``;
plumbing it through is deferred — see chat 2026-04-28.
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
        # Per-adapter prefix so Ray's log dedupe (which collapses identical
        # repeated messages) keeps these distinct across adapters.
        log_prefix = f"[multilora] ({adapter_name})"

        # Atomic save: write everything to a temp dir, then on success rename
        # to the final step_N path. The temp name does not start with ``step_``,
        # so ``find_latest_checkpoint`` ignores it even mid-write — a crash
        # leaves an orphan ``_tmp_step_N`` rather than a half-populated
        # ``step_N`` masquerading as the latest valid checkpoint.
        final_dir = config.dir / "checkpoints" / f"step_{iteration}"
        tmp_dir = config.dir / "checkpoints" / f"_tmp_step_{iteration}"
        if is_dp_rank_0:
            tmp_dir.mkdir(parents=True, exist_ok=True)
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
                native_path = tmp_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
                torch.save(shard, native_path)
                logger.info(
                    f"{log_prefix} saved Megatron shard "
                    f"({len(shard)} tensors) to {native_path}"
                )

            # ---- (2) HF PEFT format (TP-gathered, single file) ----
            # Bridge export is collective: every TP rank participates in the
            # all-gather. Only the global writer materialises the file.
            #
            # ``.contiguous().clone()`` is needed because the bridge expands
            # one Megatron fused-linear adapter into multiple HF keys that
            # share storage — the single ``linear_qkv.adapter.linear_in``
            # surfaces as ``{q,k,v}_proj.lora_A`` all aliasing the same
            # tensor, and ``linear_fc1`` similarly produces aliased
            # ``{gate,up}_proj.lora_A``. ``safetensors.save_file`` refuses
            # aliased tensors, and HF PEFT consumers expect independent
            # per-projection copies regardless.
            hf_state: dict[str, torch.Tensor] = {}
            with megatron_bridge_utils.patch_megatron_model(model):
                for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                    model, cpu=True, show_progress=False,
                ):
                    hf_state[hf_name] = weight.contiguous().clone()

        if is_global_writer:
            save_safetensors(
                hf_state,
                str(tmp_dir / "adapter_model.safetensors"),
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
            with open(tmp_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config_json, f, indent=2)
            os.sync()
            logger.info(
                f"{log_prefix} saved HF PEFT to {tmp_dir} "
                f"({len(hf_state)} tensors)"
            )

        # Wait for every rank to finish writing its part, then a single rank
        # promotes the temp dir to its final name. ``os.replace`` is atomic on
        # the same filesystem, but Linux ``rename(2)`` refuses to overwrite a
        # non-empty target dir, so we ``rmtree`` any pre-existing ``step_N``
        # first (only happens on re-saves at the same iteration).
        if dist.is_initialized():
            dist.barrier()
        if is_global_writer:
            if final_dir.exists():
                import shutil
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
            logger.info(f"{log_prefix} promoted checkpoint to {final_dir}")
        if dist.is_initialized():
            dist.barrier()


def _register_adapter(name: str, config: AdapterConfig, model) -> None:
    """Install one PENDING adapter on this rank's local model shard.

    Loads the latest cross-rank-complete checkpoint into the slot (or leaves
    construction-time values if none), then runs ``init_adapter_slot`` to
    bind ``rank``/``alpha`` and apply the rank mask. Marks ACTIVE on the
    controller. Pure model-side: the SGLang push happens inside the next
    ``update_weights()`` call.

    ``init_adapter_slot`` runs *after* ``load_adapter`` on purpose: the rank
    mask must be the source of truth for padded rows/cols. Saved shards
    can carry non-zero padded values (older code paths, mid-run rank
    changes, copy-pasted checkpoints) and silently breaking
    ``slice_lora_to_rank`` later is much worse than re-zeroing here.
    """
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    from miles.backends.megatron_utils.initialize import is_megatron_main_rank

    from ..multi_lora import find_latest_checkpoint

    log_prefix = f"[multilora] ({name})"

    ckpt_root = config.dir / "checkpoints"
    ckpt = find_latest_checkpoint(ckpt_root)
    if ckpt is None:
        logger.info(f"{log_prefix} no checkpoint under {ckpt_root}, starting from random init")
    else:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        loaded = load_adapter(model, config.slot, state_dict)
        assert loaded > 0, (
            f"{log_prefix} loaded 0 tensors from {ckpt} "
            f"(state_dict has {len(state_dict)} entries) — name mismatch?"
        )
        logger.info(f"{log_prefix} loaded from {ckpt} ({loaded} tensors)")

    init_adapter_slot(model, config.slot, rank=config.rank, alpha=config.alpha)

    if is_megatron_main_rank():
        ray.get(get_multi_lora_controller().mark_active.remote(name))
    logger.info(f"{log_prefix} installed at slot {config.slot}")


def _deregister_adapter(name: str, config: AdapterConfig, rollout_id: int, args, model, optimizer) -> None:
    """Model-side cleanup for one DRAINED adapter.

    SGLang unload happens earlier inside ``_send_multi_lora_params`` (which
    runs in the pause bracket of ``update_weights()``). This function only
    touches local trainer state: save final ckpt → clear slot → zero
    optimizer state → mark REMOVED on controller.
    """
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    from miles.backends.megatron_utils.initialize import is_megatron_main_rank

    from ..multi_lora import zero_optimizer_state_for_adapter

    log_prefix = f"[multilora] ({name})"

    save_multi_lora_checkpoints(args, model, rollout_id, {name: config})
    logger.info(f"{log_prefix} saved final checkpoint")

    clear_adapter_slot(model, config.slot)
    logger.info(f"{log_prefix} cleared adapter slot {config.slot}")

    zero_optimizer_state_for_adapter(optimizer, model, config.slot)
    optimizer.reload_model_params()

    if is_megatron_main_rank():
        ray.get(get_multi_lora_controller().mark_removed.remote(name))
    logger.info(f"{log_prefix} fully removed")


def _adapters_in_state(state):
    configs = ray.get(get_multi_lora_controller().adapter_configs.remote())
    return [(n, c) for n, c in configs.items() if c.state == state]


def load_pending_adapters(args, model, optimizer) -> int:
    """PENDING -> ACTIVE on every rank's local model shard.

    Returns the number of adapters installed so the train script knows
    whether the next ``update_weights()`` needs to push anything new to
    SGLang. Does NOT touch SGLang itself — that happens inside
    ``_send_multi_lora_params`` during ``update_weights()``.
    """
    from miles.utils.adapter_config import AdapterState

    pending = _adapters_in_state(AdapterState.PENDING)
    if not pending:
        return 0
    for name, config in pending:
        _register_adapter(name, config, model)
    optimizer.reload_model_params()
    return len(pending)


def unload_drained_adapters(args, model, optimizer, rollout_id: int) -> int:
    """DRAINED -> REMOVED model-side cleanup. Returns count for the caller's
    backuper-refresh decision.

    Caller must have already run ``update_weights()`` after the adapter went
    DRAINED so SGLang has unloaded it; otherwise SGLang holds a reference to
    a slot we're about to clear.
    """
    from miles.utils.adapter_config import AdapterState

    drained = _adapters_in_state(AdapterState.DRAINED)
    for name, config in drained:
        _deregister_adapter(name, config, rollout_id, args, model, optimizer)
    return len(drained)
