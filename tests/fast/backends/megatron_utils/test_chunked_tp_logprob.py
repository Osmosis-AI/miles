import types

import pytest
import torch

from miles.backends.megatron_utils.chunked_tp_logprob import (
    ActorOutputProjection,
    validate_chunked_tp_logprob_config,
)
from miles.backends.training_utils.loss_hub.logit_processors import (
    build_shifted_tokens_bshd,
    extract_per_sample_bshd,
)


class _OutputLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.sequence_parallel = False

    def forward(self, input_, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        return torch.nn.functional.linear(input_, weight if weight is not None else self.weight), None


class _LanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_layer = _OutputLayer()


class _NestedBridgeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()


class _Wrapper(torch.nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module


def test_install_finds_nested_bridge_output_layer():
    model = _Wrapper(_NestedBridgeModel())

    projection = ActorOutputProjection.install_on(model)

    assert projection is not None
    hidden_states = torch.tensor([[1.0, 2.0]])
    bypassed, bias = model.module.language_model.output_layer(hidden_states)
    torch.testing.assert_close(bypassed, hidden_states)
    assert bias is None
    torch.testing.assert_close(projection.linear(hidden_states), hidden_states)


def test_install_keeps_direct_output_layer_support():
    model = _LanguageModel()

    projection = ActorOutputProjection.install_on(model)

    assert projection is not None
    assert projection.output_layer is model.output_layer


def test_install_rejects_ambiguous_nested_output_layers():
    model = torch.nn.Module()
    model.first = _LanguageModel()
    model.second = _LanguageModel()

    with pytest.raises(RuntimeError, match="multiple nested actor output_layer"):
        ActorOutputProjection.install_on(model)


def _chunked_args(**overrides):
    values = {
        "allgather_cp": False,
        "chunked_tp_logprob_seq_chunk_size": 128,
        "context_parallel_size": 1,
        "enable_mtp_training": False,
        "qkv_format": "bshd",
        "train_backend": "megatron",
        "true_on_policy_mode": False,
        "use_chunked_tp_logprob_loss": True,
        "use_fused_tp_logprob_kernel": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_validate_rejects_fused_kernel_without_chunked_path():
    args = _chunked_args(use_chunked_tp_logprob_loss=False)

    with pytest.raises(ValueError, match="requires --use-chunked-tp-logprob-loss"):
        validate_chunked_tp_logprob_config(args)


def test_bshd_token_shift_and_response_extraction():
    tokens = [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22])]
    total_lengths = [4, 3]
    response_lengths = [2, 1]
    max_seq_lens = [5, 5]

    shifted = build_shifted_tokens_bshd(10, torch.device("cpu"), tokens, total_lengths, max_seq_lens)
    assert shifted.tolist() == [11, 12, 13, 0, 0, 21, 22, 0, 0, 0]

    log_probs = torch.arange(10, dtype=torch.float32)
    entropy = log_probs + 100
    log_prob_list, entropy_list = extract_per_sample_bshd(
        log_probs,
        entropy,
        total_lengths,
        response_lengths,
        max_seq_lens,
    )
    assert [item.tolist() for item in log_prob_list] == [[1.0, 2.0], [6.0]]
    assert [item.tolist() for item in entropy_list] == [[101.0, 102.0], [106.0]]
