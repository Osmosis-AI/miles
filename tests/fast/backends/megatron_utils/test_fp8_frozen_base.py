from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from miles.backends.megatron_utils.fp8_frozen_base import (
    free_frozen_base,
    is_base_linear_weight,
    quantize_frozen_base_to_fp8,
)


class _Projection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(257, 130, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = torch.nn.Module()
        self.self_attention.linear_qkv = _Projection()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.self_attention.linear_qkv(inputs)


def test_base_linear_name_filter_excludes_adapter_parameters():
    assert is_base_linear_weight("decoder.layers.0.self_attention.linear_qkv.weight")
    assert is_base_linear_weight("decoder.layers.0.mlp.experts.linear_fc1.weight1")
    assert not is_base_linear_weight("decoder.layers.0.self_attention.linear_qkv.lora_A.weight")
    assert not is_base_linear_weight("decoder.layers.0.input_layernorm.weight")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("per_layer_free", [False, True])
def test_fp8_frozen_base_roundtrip_and_lifetime(per_layer_free):
    torch.manual_seed(23)
    model = _Model()
    projection = model.self_attention.linear_qkv
    original_weight = projection.weight.detach().clone()
    inputs = torch.randn(5, 130, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference = F.linear(inputs.detach(), original_weight)

    quantize_frozen_base_to_fp8(
        [model],
        SimpleNamespace(fp8_frozen_base_per_layer_free=per_layer_free),
    )

    assert projection.weight.numel() == 0
    assert projection.fp8q_weight.dtype == torch.float8_e4m3fn
    assert projection.fp8s_weight.dtype == torch.float32

    output = model(inputs)
    normalized_max_error = (output - reference).abs().amax() / reference.abs().amax()
    assert normalized_max_error.item() < 5e-2
    assert projection.weight.numel() == original_weight.numel()

    output.float().sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    if per_layer_free:
        assert projection.weight.numel() == 0
    else:
        assert projection.weight.numel() == original_weight.numel()

    free_frozen_base([model])
    assert projection.weight.numel() == 0

    with torch.no_grad():
        model(inputs.detach())
    if per_layer_free:
        assert projection.weight.numel() == 0
    else:
        assert projection.weight.numel() == original_weight.numel()
