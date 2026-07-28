from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock


class _FakeProvider:
    def __init__(self):
        self.pre_wrap_hooks = []
        self.model = [object()]

    def finalize(self):
        self.attention_backend_at_finalize = self.attention_backend

    def register_pre_wrap_hook(self, hook):
        self.pre_wrap_hooks.append(hook)

    def provide_distributed_model(self, **_kwargs):
        return self.model


class _FakeDDPConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def finalize(self):
        self.finalized = True


def test_bridge_lora_propagates_attention_backend_before_finalize(monkeypatch):
    from megatron.bridge import AutoBridge
    from megatron.bridge.training import config as bridge_config

    from miles.backends.megatron_utils import bridge_lora_helpers
    from miles.backends.megatron_utils import lora_utils

    provider = _FakeProvider()
    bridge = SimpleNamespace(to_megatron_provider=lambda **_kwargs: provider)
    attention_backend = object()
    args = Namespace(
        hf_checkpoint="/fake/checkpoint",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        gradient_accumulation_fusion=True,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=None,
        distribute_saved_activations=False,
        attention_backend=attention_backend,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        optimizer="adam",
        accumulate_allreduce_grads_in_fp32=True,
        offload_train=False,
    )

    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", lambda *_args, **_kwargs: bridge)
    monkeypatch.setattr(bridge_config, "DistributedDataParallelConfig", _FakeDDPConfig)
    monkeypatch.setattr(
        bridge_lora_helpers,
        "load_hf_config",
        lambda _path: SimpleNamespace(architectures=["ForCausalLM"]),
    )
    monkeypatch.setattr(bridge_lora_helpers, "is_multi_lora_enabled", lambda _args: False)
    monkeypatch.setattr(lora_utils, "create_lora_instance", lambda _args: MagicMock())

    model = bridge_lora_helpers._setup_lora_model_via_bridge(args)

    assert model is provider.model
    assert provider.attention_backend is attention_backend
    assert provider.attention_backend_at_finalize is attention_backend
