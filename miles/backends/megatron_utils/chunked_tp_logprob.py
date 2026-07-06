from __future__ import annotations

import inspect
import logging
import types
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core import mpu, tensor_parallel

logger = logging.getLogger(__name__)


def should_enable_chunked_tp_logprob(args: Namespace, role: str) -> bool:
    return role == "actor" and getattr(args, "use_chunked_tp_logprob_loss", False)


def validate_chunked_tp_logprob_config(args: Namespace) -> None:
    if args.true_on_policy_mode:
        raise ValueError("--use-chunked-tp-logprob-loss does not support --true-on-policy-mode yet.")
    if args.qkv_format != "bshd":
        raise ValueError(
            f"--use-chunked-tp-logprob-loss currently supports only --qkv-format bshd. Got: {args.qkv_format}"
        )
    if args.context_parallel_size != 1:
        raise ValueError(
            "--use-chunked-tp-logprob-loss currently supports only --context-parallel-size 1. "
            f"Got: {args.context_parallel_size}"
        )
    if args.allgather_cp:
        raise ValueError("--use-chunked-tp-logprob-loss does not support --allgather-cp yet.")
    if args.chunked_tp_logprob_seq_chunk_size <= 0:
        raise ValueError(
            "--use-chunked-tp-logprob-loss requires --chunked-tp-logprob-seq-chunk-size > 0. "
            f"Got: {args.chunked_tp_logprob_seq_chunk_size}"
        )
    if getattr(args, "use_fused_tp_logprob_kernel", False) and not args.use_chunked_tp_logprob_loss:
        raise ValueError("--use-fused-tp-logprob-kernel requires --use-chunked-tp-logprob-loss to be set.")


class ActorOutputProjection:
    """Adapter over a Megatron output_layer for the chunked TP logprob bypass.

    install_on patches output_layer.forward so the actor returns hidden states.
    The loss path replays the projection in sequence chunks via linear().
    """

    def __init__(self) -> None:
        self.output_layer: torch.nn.Module | None = None
        self.original_forward = None
        self.has_runtime_gather_arg = False
        self.has_weight_arg = False
        self.weight_param_index: int | None = None
        self.runtime_weight: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.use_fused_kernel = False

    @staticmethod
    def find_output_layer(chunk: torch.nn.Module) -> torch.nn.Module | None:
        model = chunk
        while hasattr(model, "module"):
            model = model.module
        # VLM-wrapped models (e.g. Qwen3.5) nest the GPTModel under
        # language_model; get_attr_wrapped_model only unwraps .module chains
        # and raises when the attribute is absent on the wrapper.
        model = getattr(model, "language_model", model)
        return getattr(model, "output_layer", None)

    @classmethod
    def install_on(cls, model: torch.nn.Module | Sequence[torch.nn.Module]) -> ActorOutputProjection | None:
        chunks = model if isinstance(model, (list, tuple)) else [model]
        adapter: ActorOutputProjection | None = None
        patched = 0
        for chunk in chunks:
            output_layer = cls.find_output_layer(chunk)
            if output_layer is None:
                continue
            if adapter is None:
                adapter = cls()
            adapter.patch(output_layer)
            patched += 1
        if adapter is not None:
            logger.info("chunked TP logprob bypass installed on %d output_layer(s).", patched)
        return adapter

    def patch(self, output_layer: torch.nn.Module) -> None:
        if self.output_layer is None:
            self.output_layer = output_layer
            self.original_forward = output_layer.forward
            signature = inspect.signature(output_layer.forward)
            params = list(signature.parameters)
            self.has_runtime_gather_arg = "runtime_gather_output" in signature.parameters
            self.has_weight_arg = "weight" in signature.parameters
            self.weight_param_index = params.index("weight") - 1 if self.has_weight_arg else None
            self.weight = getattr(output_layer, "weight", None)
            self.bias = getattr(output_layer, "bias", None)

        adapter = self

        def passthrough(self_layer, input_: torch.Tensor, *args, **kwargs):
            if adapter.has_weight_arg:
                weight = kwargs.get("weight")
                if weight is None and args and adapter.weight_param_index is not None:
                    idx = adapter.weight_param_index
                    if 0 <= idx < len(args):
                        weight = args[idx]
                if weight is not None:
                    adapter.runtime_weight = weight
            return input_, None

        output_layer.forward = types.MethodType(passthrough, output_layer)

    @property
    def bypass_enabled(self) -> bool:
        return self.output_layer is not None

    def gather_sp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not getattr(self.output_layer, "sequence_parallel", False):
            return hidden_states
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=True
        )
        return hidden_states.transpose(0, 1).contiguous()

    def linear(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_layer = self.output_layer
        weight = self.runtime_weight if self.runtime_weight is not None else getattr(output_layer, "weight", None)

        kwargs: dict = {}
        if self.has_weight_arg and weight is not None:
            kwargs["weight"] = weight
        if self.has_runtime_gather_arg:
            kwargs["runtime_gather_output"] = False

        if weight is not None:
            hidden_states = hidden_states.to(weight.dtype)

        sequence_parallel = getattr(output_layer, "sequence_parallel", None)
        if sequence_parallel:
            output_layer.sequence_parallel = False
        try:
            output = self.original_forward(hidden_states, **kwargs)
        finally:
            if sequence_parallel is not None:
                output_layer.sequence_parallel = sequence_parallel

        if isinstance(output, tuple):
            output = output[0]
        return output

    def fused_selected_logprob(
        self,
        hidden_states: torch.Tensor,
        tokens: torch.Tensor,
        *,
        tp_group,
        rollout_temperature: float,
        with_entropy: bool,
        need_entropy_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from miles.backends.megatron_utils.kernels.selected_tp_logprob_triton import fused_selected_tp_logprob

        weight = self.runtime_weight if self.runtime_weight is not None else self.weight
        return fused_selected_tp_logprob(
            hidden_states,
            weight,
            self.bias,
            tokens,
            tp_group,
            rollout_temperature,
            with_entropy,
            need_entropy_grad,
        )


def setup_chunked_tp_logprob(model: torch.nn.Module | Sequence[torch.nn.Module], args: Namespace, role: str) -> None:
    if not should_enable_chunked_tp_logprob(args, role):
        return
    validate_chunked_tp_logprob_config(args)
    projection = ActorOutputProjection.install_on(model)
    if projection is None:
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            raise RuntimeError(
                "Requested --use-chunked-tp-logprob-loss, but no actor output_layer was found on this rank."
            )
        return
    projection.use_fused_kernel = bool(getattr(args, "use_fused_tp_logprob_kernel", False))
    if projection.use_fused_kernel:
        logger.info("fused selected-TP-logprob kernel enabled")
    args.actor_projection = projection
