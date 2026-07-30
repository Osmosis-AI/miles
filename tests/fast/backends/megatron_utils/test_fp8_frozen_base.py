from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.fp8_frozen_base import (
    free_frozen_base,
    materialized_frozen_base,
    prepare_native_fp8_frozen_base,
)


def test_native_fp8_weight_forward_backward_and_materialization():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    te = pytest.importorskip("transformer_engine.pytorch")
    available, reason = te.is_fp8_block_scaling_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    from transformer_engine.common.recipe import Float8BlockScaling

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = te.Linear(128, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
            self.proj.weight.requires_grad = False

        def forward(self, inputs):
            return self.proj(inputs)

    model = Model()
    recipe = Float8BlockScaling()
    prepare_native_fp8_frozen_base([model], recipe)
    native = model.proj.weight

    assert isinstance(native, te.Float8BlockwiseQTensor)
    assert native._columnwise_data is None

    inputs = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        model(inputs).float().sum().backward()

    assert torch.isfinite(inputs.grad).all()
    assert not model.proj._fp8_workspaces
    assert native._columnwise_data is None

    args = SimpleNamespace(fp8_frozen_base_store=True)
    with materialized_frozen_base(args, [model]):
        assert model.proj.weight.dtype == torch.bfloat16
        assert model.proj.weight is not native

    assert model.proj.weight is native
    free_frozen_base([model])
