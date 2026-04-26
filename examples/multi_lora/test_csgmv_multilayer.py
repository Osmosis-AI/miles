"""Test: csgmv vs triton through multiple LoRA layers with base_output.

The isolated kernel test showed bitwise identical results. This test
checks whether running MULTIPLE layers (where output of layer N feeds
into layer N+1) causes divergence between backends.

If csgmv diverges from triton across layers, there's a state leak
(e.g., uninitialized memory, stale permutation data) that compounds.

Usage:
    python examples/multi_lora/test_csgmv_multilayer.py
"""

import sys
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend


@dataclass
class FakeForwardBatch:
    batch_size: int
    forward_mode: ForwardMode
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: list
    extend_num_tokens: int
    lora_ids: list
    return_logprob: bool = False
    top_logprobs_nums: list = None

    def __post_init__(self):
        if self.top_logprobs_nums is None:
            self.top_logprobs_nums = [0] * self.batch_size


class FakeServerArgs:
    max_lora_chunk_size = 128


def simulate_layer(backend, x, A_weights, B_weights, base_linear_weight, slice_offsets):
    """Simulate one transformer layer: base_output = x @ W, then add LoRA."""
    base_output = x @ base_linear_weight.t()
    lora_a_out = backend.run_lora_a_sgemm(x, A_weights)
    lora_output = backend.run_lora_b_sgemm(
        x=lora_a_out,
        weights=B_weights,
        output_offset=slice_offsets,
        base_output=base_output,
    )
    return lora_output


def run_test():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    num_seqs = 8
    seq_lens = [512, 256, 1024, 128, 64, 512, 256, 768]
    num_adapters = 2
    ranks = [32, 16]
    max_rank = 32
    hidden_dim = 2560
    num_layers = 10

    weight_indices = [i % num_adapters for i in range(num_seqs)]
    scalings = [1.0, 1.0]

    # Create weights for each layer
    layer_A = []
    layer_B = []
    layer_W = []
    for _ in range(num_layers):
        A = torch.randn(num_adapters, max_rank, hidden_dim, dtype=dtype, device=device) * 0.02
        B = torch.randn(num_adapters, hidden_dim, max_rank, dtype=dtype, device=device) * 0.02
        W = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device) * 0.02
        # Zero-pad for mixed rank
        for i in range(num_adapters):
            r = ranks[i]
            if r < max_rank:
                A[i, r:, :] = 0
                B[i, :, r:] = 0
        layer_A.append(A)
        layer_B.append(B)
        layer_W.append(W)

    slice_offsets = torch.tensor([0, hidden_dim], dtype=torch.int32, device=device)

    extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    fb = FakeForwardBatch(
        batch_size=num_seqs,
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=seq_lens,
        extend_num_tokens=sum(seq_lens),
        lora_ids=list(range(num_seqs)),
    )

    # Setup backends
    triton_backend = TritonLoRABackend(max_loras_per_batch=num_adapters, device=torch.device(device))
    triton_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )

    csgmv_backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=num_adapters, device=torch.device(device),
        server_args=FakeServerArgs(),
    )
    csgmv_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )

    # Initial input
    x = torch.randn(sum(seq_lens), hidden_dim, dtype=dtype, device=device) * 0.1

    triton_x = x.clone()
    csgmv_x = x.clone()

    print(f"Running {num_layers} layers, {num_seqs} seqs, ranks={ranks}, bf16")
    print(f"Seq lens: {seq_lens}")
    print()

    for layer_idx in range(num_layers):
        triton_out = simulate_layer(
            triton_backend, triton_x,
            layer_A[layer_idx], layer_B[layer_idx], layer_W[layer_idx],
            slice_offsets,
        )
        csgmv_out = simulate_layer(
            csgmv_backend, csgmv_x,
            layer_A[layer_idx], layer_B[layer_idx], layer_W[layer_idx],
            slice_offsets,
        )

        diff = (triton_out - csgmv_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Per-adapter diff
        offset = 0
        per_adapter = {}
        for i in range(num_seqs):
            sl = seq_lens[i]
            adapter = weight_indices[i]
            d = diff[offset:offset+sl].max().item()
            per_adapter.setdefault(adapter, []).append(d)
            offset += sl

        adapter_strs = []
        for a in sorted(per_adapter):
            a_max = max(per_adapter[a])
            adapter_strs.append(f"adapter{a}={a_max:.2e}")

        print(f"  layer {layer_idx:2d}: max={max_diff:.2e}  mean={mean_diff:.2e}  {' '.join(adapter_strs)}")

        # Feed output as input to next layer
        triton_x = triton_out
        csgmv_x = csgmv_out

    torch.cuda.synchronize()

    final_diff = (triton_x - csgmv_x).abs()
    print(f"\nFinal: max={final_diff.max().item():.2e}  mean={final_diff.mean().item():.2e}")

    if final_diff.max().item() < 1e-2:
        print("PASS: backends agree through all layers")
        return True
    else:
        print("FAIL: backends diverge across layers")
        return False


def run_test_large_weights():
    """Simulate late-training conditions: large B weights that amplify errors."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(99)

    num_seqs = 8
    seq_lens = [512, 256, 1024, 128, 64, 512, 256, 768]
    num_adapters = 2
    ranks = [32, 16]
    max_rank = 32
    hidden_dim = 2560
    num_layers = 10

    weight_indices = [i % num_adapters for i in range(num_seqs)]
    scalings = [1.0, 1.0]

    slice_offsets = torch.tensor([0, hidden_dim], dtype=torch.int32, device=device)
    extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    fb = FakeForwardBatch(
        batch_size=num_seqs, forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=extend_seq_lens, extend_seq_lens_cpu=seq_lens,
        extend_num_tokens=sum(seq_lens), lora_ids=list(range(num_seqs)),
    )

    triton_backend = TritonLoRABackend(max_loras_per_batch=num_adapters, device=torch.device(device))
    triton_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )
    csgmv_backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=num_adapters, device=torch.device(device),
        server_args=FakeServerArgs(),
    )
    csgmv_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )

    print(f"\nLarge weights test (simulating step 400+): {num_layers} layers, ranks={ranks}, bf16")

    x = torch.randn(sum(seq_lens), hidden_dim, dtype=dtype, device=device) * 0.1
    triton_x, csgmv_x = x.clone(), x.clone()

    for layer_idx in range(num_layers):
        # Large B weights (simulating late training where B has grown)
        A = torch.randn(num_adapters, max_rank, hidden_dim, dtype=dtype, device=device) * 0.05
        B = torch.randn(num_adapters, hidden_dim, max_rank, dtype=dtype, device=device) * 0.5  # 25x larger
        W = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device) * 0.02
        for i in range(num_adapters):
            r = ranks[i]
            if r < max_rank:
                A[i, r:, :] = 0
                B[i, :, r:] = 0

        triton_out = simulate_layer(triton_backend, triton_x, A, B, W, slice_offsets)
        csgmv_out = simulate_layer(csgmv_backend, csgmv_x, A, B, W, slice_offsets)

        diff = (triton_out - csgmv_out).abs()
        if layer_idx % 3 == 0 or layer_idx == num_layers - 1:
            print(f"  layer {layer_idx:2d}: max={diff.max().item():.2e}  mean={diff.mean().item():.2e}")

        triton_x = triton_out
        csgmv_x = csgmv_out

    final_diff = (triton_x - csgmv_x).abs()
    print(f"  Final: max={final_diff.max().item():.2e}  mean={final_diff.mean().item():.2e}")
    return final_diff.max().item() < 1.0


def run_test_qkv_gate_up():
    """Test QKV (3 slices) and gate_up (2 slices) paths through multiple layers."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(77)

    num_seqs = 6
    seq_lens = [800, 200, 600, 100, 400, 300]
    num_adapters = 2
    ranks = [32, 16]
    max_rank = 32
    hidden_dim = 2560
    qkv_dim = 128 * 3  # q=128, k=128, v=128 per head
    gate_up_dim = 512 * 2

    weight_indices = [i % num_adapters for i in range(num_seqs)]
    scalings = [1.0, 1.0]

    extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    fb = FakeForwardBatch(
        batch_size=num_seqs, forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=extend_seq_lens, extend_seq_lens_cpu=seq_lens,
        extend_num_tokens=sum(seq_lens), lora_ids=list(range(num_seqs)),
    )

    triton_backend = TritonLoRABackend(max_loras_per_batch=num_adapters, device=torch.device(device))
    triton_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )
    csgmv_backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=num_adapters, device=torch.device(device),
        server_args=FakeServerArgs(),
    )
    csgmv_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )

    total_tokens = sum(seq_lens)

    print(f"\nQKV + gate_up test: {num_seqs} seqs, ranks={ranks}, bf16")

    # QKV test
    x = torch.randn(total_tokens, hidden_dim, dtype=dtype, device=device) * 0.1
    qkv_A = torch.randn(num_adapters, 3 * max_rank, hidden_dim, dtype=dtype, device=device) * 0.05
    qkv_B = torch.randn(num_adapters, qkv_dim, max_rank, dtype=dtype, device=device) * 0.3
    qkv_offsets = torch.tensor([0, 128, 256, 384], dtype=torch.int32, device=device)
    for i in range(num_adapters):
        r = ranks[i]
        if r < max_rank:
            for s in range(3):
                qkv_A[i, s * max_rank + r:(s+1) * max_rank, :] = 0
            qkv_B[i, :, r:] = 0

    triton_qkv = triton_backend.run_qkv_lora(x=x, qkv_lora_a=qkv_A, qkv_lora_b=qkv_B,
                                                output_offset=qkv_offsets, max_qkv_out_dim=128)
    csgmv_qkv = csgmv_backend.run_qkv_lora(x=x, qkv_lora_a=qkv_A, qkv_lora_b=qkv_B,
                                              output_offset=qkv_offsets, max_qkv_out_dim=128)
    qkv_diff = (triton_qkv - csgmv_qkv).abs()
    print(f"  QKV:     max={qkv_diff.max().item():.2e}  mean={qkv_diff.mean().item():.2e}")

    # gate_up test
    gu_A = torch.randn(num_adapters, 2 * max_rank, hidden_dim, dtype=dtype, device=device) * 0.05
    gu_B = torch.randn(num_adapters, gate_up_dim, max_rank, dtype=dtype, device=device) * 0.3
    gu_offsets = torch.tensor([0, 512, 1024], dtype=torch.int32, device=device)
    for i in range(num_adapters):
        r = ranks[i]
        if r < max_rank:
            for s in range(2):
                gu_A[i, s * max_rank + r:(s+1) * max_rank, :] = 0
            gu_B[i, :, r:] = 0

    triton_gu = triton_backend.run_gate_up_lora(x=x, gate_up_lora_a=gu_A, gate_up_lora_b=gu_B)
    csgmv_gu = csgmv_backend.run_gate_up_lora(x=x, gate_up_lora_a=gu_A, gate_up_lora_b=gu_B)
    gu_diff = (triton_gu - csgmv_gu).abs()
    print(f"  gate_up: max={gu_diff.max().item():.2e}  mean={gu_diff.mean().item():.2e}")

    ok = qkv_diff.max().item() < 0.01 and gu_diff.max().item() < 0.01
    return ok


def run_test_decode():
    """Test decode path (1 token per sequence) — this is the generation hot path."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(55)

    num_seqs = 32  # realistic decode batch
    num_adapters = 2
    ranks = [32, 16]
    max_rank = 32
    hidden_dim = 2560

    weight_indices = [i % num_adapters for i in range(num_seqs)]
    scalings = [1.0, 1.0]

    # Decode: 1 token per sequence
    seq_lens = [1] * num_seqs
    extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    fb = FakeForwardBatch(
        batch_size=num_seqs, forward_mode=ForwardMode.DECODE,
        extend_seq_lens=extend_seq_lens, extend_seq_lens_cpu=seq_lens,
        extend_num_tokens=num_seqs, lora_ids=list(range(num_seqs)),
    )

    triton_backend = TritonLoRABackend(max_loras_per_batch=num_adapters, device=torch.device(device))
    triton_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )
    csgmv_backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=num_adapters, device=torch.device(device),
        server_args=FakeServerArgs(),
    )
    csgmv_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )

    slice_offsets = torch.tensor([0, hidden_dim], dtype=torch.int32, device=device)
    print(f"\nDecode test: bs={num_seqs}, ranks={ranks}, bf16")

    # Run 100 decode steps (simulating generation)
    x = torch.randn(num_seqs, hidden_dim, dtype=dtype, device=device) * 0.1
    triton_x, csgmv_x = x.clone(), x.clone()

    max_diffs = []
    for step in range(100):
        A = torch.randn(num_adapters, max_rank, hidden_dim, dtype=dtype, device=device) * 0.05
        B = torch.randn(num_adapters, hidden_dim, max_rank, dtype=dtype, device=device) * 0.3
        W = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device) * 0.02
        for i in range(num_adapters):
            r = ranks[i]
            if r < max_rank:
                A[i, r:, :] = 0
                B[i, :, r:] = 0

        triton_out = simulate_layer(triton_backend, triton_x, A, B, W, slice_offsets)
        csgmv_out = simulate_layer(csgmv_backend, csgmv_x, A, B, W, slice_offsets)
        max_diffs.append((triton_out - csgmv_out).abs().max().item())
        triton_x = triton_out
        csgmv_x = csgmv_out

    print(f"  step   0: max_diff={max_diffs[0]:.2e}")
    print(f"  step  49: max_diff={max_diffs[49]:.2e}")
    print(f"  step  99: max_diff={max_diffs[99]:.2e}")
    print(f"  trend: {'growing' if max_diffs[99] > max_diffs[0] * 5 else 'stable'}")
    return max_diffs[99] < 1.0


def run_test_skewed_adapters():
    """Highly skewed adapter distribution: 1 seq on adapter 0, 7 seqs on adapter 1."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(123)

    num_seqs = 8
    seq_lens = [2048, 64, 64, 64, 64, 64, 64, 64]
    num_adapters = 2
    ranks = [32, 16]
    max_rank = 32
    hidden_dim = 2560

    # 1 long seq on adapter 0, 7 short seqs on adapter 1
    weight_indices = [0] + [1] * 7
    scalings = [1.0, 1.0]
    slice_offsets = torch.tensor([0, hidden_dim], dtype=torch.int32, device=device)

    extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    fb = FakeForwardBatch(
        batch_size=num_seqs, forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=extend_seq_lens, extend_seq_lens_cpu=seq_lens,
        extend_num_tokens=sum(seq_lens), lora_ids=list(range(num_seqs)),
    )

    triton_backend = TritonLoRABackend(max_loras_per_batch=num_adapters, device=torch.device(device))
    triton_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )
    csgmv_backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=num_adapters, device=torch.device(device),
        server_args=FakeServerArgs(),
    )
    csgmv_backend.prepare_lora_batch(
        forward_batch=fb, weight_indices=weight_indices,
        lora_ranks=ranks, scalings=scalings, use_cuda_graph=False,
    )

    print(f"\nSkewed adapter test: 1 long seq (adapter 0) + 7 short (adapter 1)")

    x = torch.randn(sum(seq_lens), hidden_dim, dtype=dtype, device=device) * 0.1

    A = torch.randn(num_adapters, max_rank, hidden_dim, dtype=dtype, device=device) * 0.05
    B = torch.randn(num_adapters, hidden_dim, max_rank, dtype=dtype, device=device) * 0.5
    W = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device) * 0.02
    for i in range(num_adapters):
        r = ranks[i]
        if r < max_rank:
            A[i, r:, :] = 0
            B[i, :, r:] = 0

    triton_out = simulate_layer(triton_backend, x, A, B, W, slice_offsets)
    csgmv_out = simulate_layer(csgmv_backend, x, A, B, W, slice_offsets)

    diff = (triton_out - csgmv_out).abs()

    # Per-adapter breakdown
    adapter0_diff = diff[:seq_lens[0]].max().item()
    adapter1_diff = diff[seq_lens[0]:].max().item()

    print(f"  adapter 0 (long):  max={adapter0_diff:.2e}")
    print(f"  adapter 1 (short): max={adapter1_diff:.2e}")
    print(f"  total:             max={diff.max().item():.2e}  mean={diff.mean().item():.2e}")
    return diff.max().item() < 0.01


if __name__ == "__main__":
    print("=" * 70)
    print("csgmv vs triton: multi-layer + pathological cases")
    print("=" * 70)

    results = [
        ("Multi-layer (10 layers)", run_test),
        ("Large B weights (late training)", run_test_large_weights),
        ("QKV + gate_up paths", run_test_qkv_gate_up),
        ("Decode path (100 steps)", run_test_decode),
        ("Skewed adapter distribution", run_test_skewed_adapters),
    ]

    outcomes = []
    for name, fn in results:
        try:
            ok = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok = False
        outcomes.append((name, ok))

    print("\n" + "=" * 70)
    for name, ok in outcomes:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 70)
    sys.exit(0 if all(ok for _, ok in outcomes) else 1)
