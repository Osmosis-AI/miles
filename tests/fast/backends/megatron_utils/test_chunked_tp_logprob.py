import types

import pytest
import torch

from miles.backends.megatron_utils.chunked_tp_logprob import setup_chunked_tp_logprob
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy


def _args(**overrides):
    values = dict(
        actor_projection=None,
        allgather_cp=False,
        enable_mtp_training=False,
        log_probs_chunk_size=2,
        qkv_format="bshd",
        recompute_chunked_tp_logprob_loss=False,
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        use_chunked_tp_logprob_loss=True,
        vocab_size=3,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _patch_math(monkeypatch, calculate, *, cp_rank=0, cp_size=1):
    from miles.backends.training_utils import cp_utils
    from miles.backends.training_utils.loss_hub import logit_processors

    state = types.SimpleNamespace(
        cp=types.SimpleNamespace(rank=cp_rank, size=cp_size, group=None),
        tp=types.SimpleNamespace(rank=0, size=1, group=None),
    )
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)
    monkeypatch.setattr(logit_processors, "calculate_log_probs_and_entropy", calculate)


def test_setup_bypasses_nested_output_layer_and_reuses_tied_weight():
    class OutputLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("weight", None)
            self.sequence_parallel = False

        def forward(self, hidden, weight=None, runtime_gather_output=None):
            return torch.nn.functional.linear(hidden, weight), None

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.output_layer = OutputLayer()
    args = _args(log_probs_chunk_size=-1)
    setup_chunked_tp_logprob([model], args, "actor")

    hidden = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    bypassed, _ = model.language_model.output_layer(hidden, weight=weight)

    torch.testing.assert_close(bypassed, hidden)
    torch.testing.assert_close(args.actor_projection(hidden), torch.tensor([[2.0, 6.0]]))
    assert args.log_probs_chunk_size == 256


def test_response_hidden_states_are_packed_and_chunked(monkeypatch):
    def calculate(logits, tokens, _group, **kwargs):
        assert logits.dtype == torch.bfloat16
        return logits[:, 0].float() + tokens, logits[:, 0].float() if kwargs["with_entropy"] else None

    _patch_math(monkeypatch, calculate)

    class Projection:
        def __init__(self):
            self.calls = []

        def gather_sequence_parallel(self, hidden):
            return hidden

        def __call__(self, hidden):
            self.calls.append(hidden.size(0))
            return hidden

    projection = Projection()
    args = _args(actor_projection=projection, log_probs_chunk_size=3)
    result = get_log_probs_and_entropy(
        torch.arange(8, dtype=torch.bfloat16).view(1, 8, 1),
        args=args,
        unconcat_tokens=[torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])],
        total_lengths=[3, 3],
        response_lengths=[2, 2],
        with_entropy=True,
        max_seq_lens=[4, 4],
    )

    assert projection.calls == [3, 1]
    assert [values.tolist() for values in result["log_probs"]] == [[1.0, 3.0], [5.0, 7.0]]
    assert [values.tolist() for values in result["entropy"]] == [[0.0, 1.0], [4.0, 5.0]]


@pytest.mark.parametrize(
    ("with_entropy", "entropy_requires_grad"),
    [(False, False), (True, False), (True, True)],
)
def test_chunk_recompute_preserves_outputs_and_gradients(monkeypatch, with_entropy, entropy_requires_grad):
    def calculate(logits, tokens, _group, **kwargs):
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, tokens[:, None]).squeeze(-1)
        entropy_log_probs = log_probs if kwargs["entropy_requires_grad"] else log_probs.detach()
        entropy = -(entropy_log_probs.exp() * entropy_log_probs).sum(-1) if kwargs["with_entropy"] else None
        return selected, entropy

    _patch_math(monkeypatch, calculate)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
            self.calls = 0

        def gather_sequence_parallel(self, hidden):
            return hidden

        def forward(self, hidden):
            self.calls += 1
            return hidden @ self.weight.T

    def run(recompute):
        projection = Projection()
        hidden = torch.arange(10, dtype=torch.float32).view(1, 5, 2).requires_grad_()
        result = get_log_probs_and_entropy(
            hidden,
            args=_args(actor_projection=projection, recompute_chunked_tp_logprob_loss=recompute),
            unconcat_tokens=[torch.tensor([0, 1, 2, 0, 1])],
            total_lengths=[5],
            response_lengths=[4],
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=[5],
        )
        loss = result["log_probs"][0].sum()
        if with_entropy:
            loss = loss + result["entropy"][0].sum()
        forward_calls = projection.calls
        loss.backward()
        return result, hidden.grad, projection.weight.grad, forward_calls, projection.calls

    baseline = run(False)
    recomputed = run(True)
    torch.testing.assert_close(recomputed[0]["log_probs"][0], baseline[0]["log_probs"][0])
    if with_entropy:
        torch.testing.assert_close(recomputed[0]["entropy"][0], baseline[0]["entropy"][0])
    torch.testing.assert_close(recomputed[1], baseline[1])
    torch.testing.assert_close(recomputed[2], baseline[2])
    assert baseline[3:] == (2, 2)
    assert recomputed[3:] == (2, 4)


@pytest.mark.parametrize(
    ("qkv_format", "cp_rank", "positions"),
    [
        ("bshd", 0, [0, 1, 6, 7]),
        ("bshd", 1, [2, 3, 4, 5]),
        ("thd", 0, [0, 1, 6, 7]),
        ("thd", 1, [2, 3, 4, 5]),
    ],
)
def test_context_parallel_response_slicing(monkeypatch, qkv_format, cp_rank, positions):
    seen_tokens = []

    def calculate(logits, tokens, _group, **_kwargs):
        seen_tokens.extend(tokens.tolist())
        return logits[:, 0], None

    _patch_math(monkeypatch, calculate, cp_rank=cp_rank, cp_size=2)

    class Projection:
        gather_sequence_parallel = __call__ = staticmethod(lambda hidden: hidden)

    result = get_log_probs_and_entropy(
        torch.tensor(positions, dtype=torch.float32).view(1, 4, 1),
        args=_args(actor_projection=Projection(), qkv_format=qkv_format),
        unconcat_tokens=[torch.arange(8)],
        total_lengths=[8],
        response_lengths=[6],
        max_seq_lens=[8] if qkv_format == "bshd" else None,
    )

    expected_positions = [1, 6] if cp_rank == 0 else [2, 3, 4, 5]
    expected_tokens = [2, 7] if cp_rank == 0 else [3, 4, 5, 6]
    assert result["log_probs"][0].tolist() == expected_positions
    assert seen_tokens == expected_tokens
