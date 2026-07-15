import logging

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora_utils import _is_adapter_param_name
from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton

from tools.fp8_cast_bf16 import weight_dequant

logger = logging.getLogger(__name__)

# TODO: TE-native fp8 params (Float8BlockScaling) so GEMMs consume the stored fp8
#       directly and the bf16 copy never exists.

BLOCK = 128

# Frozen base linear weights to store as block-fp8 (mirrors the export-side
# allowlist in megatron_to_hf/processors/quantizer_fp8.py, plus GDN in/out proj).
LINEAR_SUBSTR = (
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "self_attention.linear_q_proj",
    "self_attention.linear_q_down_proj",
    "self_attention.linear_q_up_proj",
    "self_attention.linear_kv_down_proj",
    "self_attention.linear_kv_up_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    "mlp.shared_experts.linear_fc1",
    "mlp.shared_experts.linear_fc2",
    "mlp.experts.linear_fc1",
    "mlp.experts.linear_fc2",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    # bridge-built GDN layers keep the GatedDeltaNet module at self_attention
    "self_attention.in_proj",
    "self_attention.out_proj",
)


def rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def is_base_linear_weight(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    if not (leaf == "weight" or (leaf.startswith("weight") and leaf[len("weight") :].isdigit())):
        return False
    if _is_adapter_param_name(name):
        return False
    return any(s in name for s in LINEAR_SUBSTR)


def should_quantize(name: str, param: torch.nn.Parameter) -> bool:
    return param.dim() == 2 and not param.requires_grad and is_base_linear_weight(name)


def family(name: str) -> str:
    if "mlp.experts." in name:
        return "moe_experts"
    if "shared_experts" in name:
        return "shared_experts"
    if "linear_attn" in name or "self_attention.in_proj" in name or "self_attention.out_proj" in name:
        return "gdn"
    if "self_attention" in name:
        return "attention"
    if "mlp.linear_fc" in name:
        return "dense_mlp"
    return "other"


def install_fp8_hooks(module: torch.nn.Module, per_layer_free: bool) -> None:
    def pre(mod, inputs):
        for leaf, (param, shape, dtype) in mod.fp8_frozen_entries.items():
            if param.data.numel():
                continue
            q = getattr(mod, f"fp8q_{leaf}")
            s = getattr(mod, f"fp8s_{leaf}")
            param.data = weight_dequant(q, s, BLOCK).to(dtype).reshape(shape)

    module.register_forward_pre_hook(pre)

    if not per_layer_free:
        # bf16 stays until free_frozen_base at offload: TE grouped backward and
        # activation recompute read self.weight after the forward.
        return

    def free_entries(mod):
        for param, _, dtype in mod.fp8_frozen_entries.values():
            param.data = torch.empty(0, dtype=dtype, device=param.data.device)

    def post(mod, inputs, output):
        # no_grad (checkpointed outer forward, forward-only): nothing saves the weight,
        # free now — the pre-hook re-materializes on recompute. Under grad, free after
        # the module's backward via its grad_fn post-hook (TE forward = one autograd
        # Function; frozen base has no wgrad). multi_grad_hook cannot be used here:
        # its closure↔graph reference cycle retains every microbatch graph.
        if not torch.is_grad_enabled():
            free_entries(mod)
            return
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor) and out.grad_fn is not None:
            out.grad_fn.register_hook(lambda grad_inputs, grad_outputs: free_entries(mod))

    module.register_forward_hook(post)


def frozen_fp8_param_ids(model_chunks) -> set[int]:
    return {
        id(param)
        for chunk in model_chunks
        for module in chunk.modules()
        for param, _, _ in getattr(module, "fp8_frozen_entries", {}).values()
    }


def free_frozen_base(model_chunks) -> None:
    """Drop the transient bf16 base so only the fp8 buffers stay resident; call post-backward."""
    for chunk in model_chunks:
        for module in chunk.modules():
            for param, _, dtype in getattr(module, "fp8_frozen_entries", {}).values():
                param.data = torch.empty(0, dtype=dtype, device=param.data.device)


def quantize_frozen_base_to_fp8(model_chunks, args) -> None:
    freed_bytes = 0
    counts: dict[str, int] = {}
    roundtrip_relerr = None

    for chunk in model_chunks:
        for mod_name, module in chunk.named_modules():
            entries: dict = {}
            for leaf, param in list(module.named_parameters(recurse=False)):
                full = f"{mod_name}.{leaf}" if mod_name else leaf
                if not should_quantize(full, param):
                    continue

                w = param.data.contiguous()
                q, s = blockwise_cast_to_fp8_triton(w, [BLOCK, BLOCK])

                if roundtrip_relerr is None:
                    recon = weight_dequant(q.contiguous(), s.contiguous(), BLOCK).to(w.dtype)
                    denom = w.abs().amax().clamp(min=1e-6)
                    roundtrip_relerr = ((recon - w).abs().amax() / denom).item()

                module.register_buffer(f"fp8q_{leaf}", q, persistent=False)
                module.register_buffer(f"fp8s_{leaf}", s, persistent=False)
                entries[leaf] = (param, tuple(param.shape), param.dtype)

                freed_bytes += w.numel() * w.element_size()
                freed_bytes -= q.numel() * q.element_size() + s.numel() * s.element_size()
                counts[family(full)] = counts.get(family(full), 0) + 1
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)

            if entries:
                module.fp8_frozen_entries = entries
                install_fp8_hooks(module, args.fp8_frozen_base_per_layer_free)

    if rank0() and counts:
        logger.info(
            "fp8 frozen base: quantized %d tensors %s, freed ~%.2f GB/rank, roundtrip_relerr=%.2e",
            sum(counts.values()),
            counts,
            freed_bytes / (1024**3),
            roundtrip_relerr,
        )
