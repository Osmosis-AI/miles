from __future__ import annotations

import types
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core import tensor_parallel

from miles.backends.training_utils.parallel import get_parallel_state


def _find_output_layer(model: Sequence[torch.nn.Module]) -> torch.nn.Module | None:
    layers = {
        id(module): module
        for chunk in model
        for name, module in chunk.named_modules()
        if name.rsplit(".", 1)[-1] == "output_layer"
    }
    if len(layers) > 1:
        raise RuntimeError("Chunked log-probs require exactly one actor output layer.")
    return next(iter(layers.values()), None)


class ChunkedOutputProjection:
    """Return final hidden states from the model, then replay its LM head in chunks."""

    def __init__(self, output_layer: torch.nn.Module) -> None:
        self.output_layer = output_layer
        self.forward = output_layer.forward
        self.runtime_weight: torch.Tensor | None = None
        projection = self

        def return_hidden(_layer, hidden_states: torch.Tensor, weight=None, *_args, **_kwargs):
            if weight is not None:
                projection.runtime_weight = weight
            return hidden_states, None

        output_layer.forward = types.MethodType(return_hidden, output_layer)

    def gather_sequence_parallel(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not getattr(self.output_layer, "sequence_parallel", False):
            return hidden_states
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=True,
        )
        return hidden_states.transpose(0, 1).contiguous()

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.runtime_weight if self.runtime_weight is not None else self.output_layer.weight
        sequence_parallel = self.output_layer.sequence_parallel
        self.output_layer.sequence_parallel = False
        try:
            logits, _ = self.forward(
                hidden_states.to(weight.dtype),
                weight=weight,
                runtime_gather_output=False,
            )
        finally:
            self.output_layer.sequence_parallel = sequence_parallel
        return logits


def setup_chunked_tp_logprob(model: Sequence[torch.nn.Module], args: Namespace, role: str) -> None:
    if role != "actor" or not getattr(args, "use_chunked_tp_logprob_loss", False):
        return
    if args.true_on_policy_mode or args.enable_mtp_training or args.allgather_cp:
        raise ValueError("Chunked log-probs do not support true-on-policy, MTP, or all-gather CP.")
    if args.log_probs_chunk_size == -1:
        args.log_probs_chunk_size = 256
    elif args.log_probs_chunk_size <= 0:
        raise ValueError("--log-probs-chunk-size must be positive.")
    output_layer = _find_output_layer(model)
    if output_layer is None:
        if get_parallel_state().is_pp_last_stage:
            raise RuntimeError("No actor output layer found for chunked log-probs.")
        return
    args.actor_projection = ChunkedOutputProjection(output_layer)
