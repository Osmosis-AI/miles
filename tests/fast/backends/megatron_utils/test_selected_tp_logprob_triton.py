import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from miles.backends.megatron_utils.kernels.selected_tp_logprob_triton import fused_selected_tp_logprob


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_selected_logprob_matches_reference_forward_and_backward(with_bias):
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    temperature = 0.73

    hidden_ref = torch.randn(7, 32, device=device, dtype=dtype, requires_grad=True)
    weight_ref = torch.randn(513, 32, device=device, dtype=dtype, requires_grad=True)
    bias_ref = torch.randn(513, device=device, dtype=dtype, requires_grad=True) if with_bias else None
    tokens = torch.randint(0, 513, (7,), device=device)
    grad_log_prob = torch.randn(7, device=device)
    grad_entropy = torch.randn(7, device=device)

    logits = F.linear(hidden_ref, weight_ref, bias_ref).float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_ref = log_probs.gather(-1, tokens[:, None]).squeeze(-1)
    entropy_ref = -(log_probs.exp() * log_probs).sum(dim=-1)
    reference_loss = (selected_ref * grad_log_prob + entropy_ref * grad_entropy).sum()
    reference_loss.backward()

    hidden = hidden_ref.detach().clone().requires_grad_(True)
    weight = weight_ref.detach().clone().requires_grad_(True)
    bias = bias_ref.detach().clone().requires_grad_(True) if bias_ref is not None else None
    selected, entropy = fused_selected_tp_logprob(
        hidden,
        weight,
        bias,
        tokens,
        tp_group=None,
        rollout_temperature=temperature,
        with_entropy=True,
        need_entropy_grad=True,
    )
    fused_loss = (selected * grad_log_prob + entropy * grad_entropy).sum()
    fused_loss.backward()

    torch.testing.assert_close(selected, selected_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(entropy, entropy_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=5e-2, atol=5e-2)
    if with_bias:
        torch.testing.assert_close(bias.grad, bias_ref.grad, rtol=5e-2, atol=5e-2)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _distributed_parity_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/selected-tp-logprob-triton-rank-{rank}"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(19)
        device = torch.device("cuda", rank)
        dtype = torch.bfloat16
        temperature = 0.83
        local_vocab_size = 257
        global_vocab_size = local_vocab_size * world_size

        hidden_base = torch.randn(7, 32, device=device, dtype=dtype)
        weight_base = torch.randn(global_vocab_size, 32, device=device, dtype=dtype)
        bias_base = torch.randn(global_vocab_size, device=device, dtype=dtype)
        tokens = torch.randint(0, global_vocab_size, (7,), device=device)
        grad_log_prob = torch.randn(7, device=device)
        grad_entropy = torch.randn(7, device=device)

        for with_bias in (False, True):
            hidden_ref = hidden_base.detach().clone().requires_grad_(True)
            weight_ref = weight_base.detach().clone().requires_grad_(True)
            bias_ref = bias_base.detach().clone().requires_grad_(True) if with_bias else None
            logits = F.linear(hidden_ref, weight_ref, bias_ref).float() / temperature
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_ref = log_probs.gather(-1, tokens[:, None]).squeeze(-1)
            entropy_ref = -(log_probs.exp() * log_probs).sum(dim=-1)
            reference_loss = (selected_ref * grad_log_prob + entropy_ref * grad_entropy).sum()
            reference_loss.backward()

            vocab_start = rank * local_vocab_size
            vocab_end = vocab_start + local_vocab_size
            hidden = hidden_base.detach().clone().requires_grad_(True)
            weight = weight_base[vocab_start:vocab_end].detach().clone().requires_grad_(True)
            bias = (
                bias_base[vocab_start:vocab_end].detach().clone().requires_grad_(True)
                if with_bias
                else None
            )
            selected, entropy = fused_selected_tp_logprob(
                hidden,
                weight,
                bias,
                tokens,
                tp_group=dist.group.WORLD,
                rollout_temperature=temperature,
                with_entropy=True,
                need_entropy_grad=True,
            )
            fused_loss = (selected * grad_log_prob + entropy * grad_entropy).sum()
            fused_loss.backward()

            torch.testing.assert_close(selected, selected_ref, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(entropy, entropy_ref, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(
                weight.grad,
                weight_ref.grad[vocab_start:vocab_end],
                rtol=5e-2,
                atol=5e-2,
            )
            if with_bias:
                torch.testing.assert_close(
                    bias.grad,
                    bias_ref.grad[vocab_start:vocab_end],
                    rtol=5e-2,
                    atol=5e-2,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_fused_selected_logprob_matches_two_rank_tp_reference():
    world_size = 2
    mp.spawn(
        _distributed_parity_worker,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )
