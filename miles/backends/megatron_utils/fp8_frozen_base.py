import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _trim(module) -> None:
    for weight, _dtype in getattr(module, "_fp8_frozen_weights", {}).values():
        weight._quantizer.set_usage(rowwise=True, columnwise=False)
        weight.update_usage(rowwise_usage=True, columnwise_usage=False)


def _install_release_hook(module) -> None:
    def after_forward(mod, inputs, _output):
        if not torch.is_grad_enabled():
            _trim(mod)
            return
        tensor = next((value for value in inputs if isinstance(value, torch.Tensor) and value.requires_grad), None)
        if tensor is None:
            _trim(mod)
            return

        mod._fp8_pending_backwards = getattr(mod, "_fp8_pending_backwards", 0) + 1

        def release(grad):
            mod._fp8_pending_backwards -= 1
            if mod._fp8_pending_backwards == 0:
                _trim(mod)
            return grad

        tensor.register_hook(release)

    module.register_forward_hook(after_forward)


@torch.no_grad()
def prepare_native_fp8_frozen_base(model_chunks, recipe) -> None:
    """Replace frozen TE linear weights with native block-FP8 parameters before DDP."""
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8BlockScaling
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        raise RuntimeError(f"Native block FP8 is unavailable: {reason}")
    if not isinstance(recipe, Float8BlockScaling) or recipe.w_block_scaling_dim != 2:
        raise RuntimeError("--fp8-frozen-base-store requires TE 128x128 block scaling")

    qparams = recipe.fp8_quant_fwd_weight
    template = te.Float8BlockQuantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        amax_epsilon=qparams.amax_epsilon,
        force_pow_2_scales=qparams.power_2_scale,
        block_scaling_dim=recipe.w_block_scaling_dim,
    )
    template.internal = False

    selected = []
    for chunk in model_chunks:
        for module_name, module in chunk.named_modules():
            if not isinstance(module, TransformerEngineBaseModule):
                continue
            for name, weight in module.named_parameters(recurse=False):
                if (
                    name.startswith("weight")
                    and weight.ndim == 2
                    and not weight.requires_grad
                    and not getattr(weight, "shared", False)
                ):
                    selected.append((module_name, module, name, weight))

    invalid = [
        f"{module_name}.{name}{tuple(weight.shape)}"
        for module_name, _module, name, weight in selected
        if weight.device.type != "cuda" or not template.is_quantizable(weight)
    ]
    if not selected or invalid:
        raise RuntimeError(f"Native FP8 found no usable frozen TE weights; invalid={invalid[:8]}")

    bf16_bytes = 0
    fp8_bytes = 0
    for _module_name, module, name, weight in selected:
        quantized = template.copy()(weight.detach().contiguous())
        native = torch.nn.Parameter(quantized, requires_grad=False)
        native.__dict__.update(weight.__dict__)
        native._fp8_frozen_base = True
        module.register_parameter(name, native)
        module._fp8_frozen_weights = getattr(module, "_fp8_frozen_weights", {})
        module._fp8_frozen_weights[name] = (native, weight.dtype)
        module.primary_weights_in_fp8 = True
        if hasattr(module, "_fp8_workspaces"):
            module._fp8_workspaces.clear()
        if not getattr(module, "_fp8_release_hook_installed", False):
            _install_release_hook(module)
            module._fp8_release_hook_installed = True
        bf16_bytes += weight.numel() * weight.element_size()
        fp8_bytes += native._rowwise_data.nbytes + native._rowwise_scale_inv.nbytes

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            "Stored %d frozen TE weights in native FP8 (%.2f GiB/rank saved)",
            len(selected),
            (bf16_bytes - fp8_bytes) / 2**30,
        )


def free_frozen_base(model_chunks) -> None:
    for chunk in model_chunks:
        for module in chunk.modules():
            module._fp8_pending_backwards = 0
            _trim(module)


@contextmanager
def materialized_frozen_base(args, model_chunks):
    """Temporarily expose native FP8 weights as BF16 for Bridge load/export."""
    if not getattr(args, "fp8_frozen_base_store", False):
        yield
        return

    swapped = []
    try:
        for chunk in model_chunks:
            for module in chunk.modules():
                for name, (native, dtype) in getattr(module, "_fp8_frozen_weights", {}).items():
                    if module._parameters[name] is not native:
                        continue
                    weight = torch.nn.Parameter(native.dequantize(dtype=dtype), requires_grad=False)
                    weight.__dict__.update(native.__dict__)
                    module.register_parameter(name, weight)
                    module.primary_weights_in_fp8 = False
                    swapped.append((module, name, native))
        yield
    finally:
        for module, name, native in reversed(swapped):
            module.register_parameter(name, native)
            module.primary_weights_in_fp8 = True
            if hasattr(module, "_fp8_workspaces"):
                module._fp8_workspaces.clear()
