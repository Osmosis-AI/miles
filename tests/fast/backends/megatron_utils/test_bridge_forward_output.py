from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from miles.backends.megatron_utils.bridge_lora_helpers import _setup_lora_model_via_bridge
from miles.backends.megatron_utils.model_provider import _wrap_bridge_forward_primary_output


class _Model:
    def __init__(self, output):
        self.output = output

    def forward(self, *args, **kwargs):
        return self.output


def test_wrap_bridge_forward_primary_output_unwraps_tuple():
    logits = torch.tensor([1.0])
    model = _wrap_bridge_forward_primary_output(_Model((logits, None)))

    assert model.forward() is logits


def test_wrap_bridge_forward_primary_output_preserves_non_tuple():
    logits = torch.tensor([1.0])
    metadata = {"hidden_states": logits}

    tensor_model = _wrap_bridge_forward_primary_output(_Model(logits))
    metadata_model = _wrap_bridge_forward_primary_output(_Model(metadata))

    assert tensor_model.forward() is logits
    assert metadata_model.forward() is metadata


def test_wrap_bridge_forward_primary_output_wraps_every_chunk():
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    chunks = [_Model((first, "first-mask")), _Model((second, "second-mask"))]

    result = _wrap_bridge_forward_primary_output(chunks)

    assert result is chunks
    assert chunks[0].forward() is first
    assert chunks[1].forward() is second


def test_lora_bridge_installs_primary_output_hook_after_lora():
    provider = MagicMock()
    hooks = []
    provider.register_pre_wrap_hook.side_effect = hooks.append

    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    chunks = [_Model((first, "first-mask")), _Model((second, "second-mask"))]

    def provide_distributed_model(**_kwargs):
        transformed = chunks
        for hook in hooks:
            transformed = hook(transformed)
        return transformed

    provider.provide_distributed_model.side_effect = provide_distributed_model
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider
    lora = MagicMock(side_effect=lambda model_chunks, *, training: model_chunks)
    args = Namespace(
        hf_checkpoint="model",
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=2,
        gradient_accumulation_fusion=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=None,
        distribute_saved_activations=False,
        attention_backend="flash",
        target_modules=[],
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        dsa_attention_backend="megatron",
        optimizer="adam",
        accumulate_allreduce_grads_in_fp32=True,
        offload_train=False,
    )

    with (
        patch("megatron.bridge.AutoBridge.from_hf_pretrained", return_value=bridge),
        patch(
            "megatron.bridge.training.config.DistributedDataParallelConfig",
            return_value=MagicMock(),
        ),
        patch(
            "miles.backends.megatron_utils.bridge_lora_helpers.load_hf_config",
            return_value=SimpleNamespace(architectures=["ForCausalLM"]),
        ),
        patch(
            "miles.backends.megatron_utils.bridge_lora_helpers.is_multi_lora_enabled",
            return_value=False,
        ),
        patch(
            "miles.backends.megatron_utils.lora_utils.create_lora_instance",
            return_value=lora,
        ),
    ):
        result = _setup_lora_model_via_bridge(args)

    assert hooks[-1] is _wrap_bridge_forward_primary_output
    assert result is chunks
    assert chunks[0].forward() is first
    assert chunks[1].forward() is second
