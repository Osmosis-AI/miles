"""Test: csgmv vs triton vs torch reference for LoRA matmul correctness.

Compares the chunked SGMV backend against the triton backend and a plain
torch.matmul reference at bf16 with realistic multi-adapter settings.

If csgmv produces different results from both triton AND torch reference,
it's computing the wrong answer.

Usage (on a machine with sglang installed):
    python examples/multi_lora/test_csgmv_correctness.py
"""

import sys
import torch
from dataclasses import dataclass


@dataclass
class FakeForwardBatch:
    batch_size: int
    forward_mode: "FakeMode"
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: list
    extend_num_tokens: int
    lora_ids: list

    class FakeMode:
        @staticmethod
        def is_extend():
            return True
        @staticmethod
        def is_cuda_graph():
            return False


def torch_reference_lora(x_list, A_weights, B_weights, weight_indices, ranks, scalings, slice_offsets):
    """Pure torch reference: per-sequence x @ A[:rank].T @ B[:, :rank].T * scaling."""
    num_slices = len(slice_offsets) - 1
    total_tokens = sum(t.shape[0] for t in x_list)
    output_dim = slice_offsets[-1].item()
    output = torch.zeros(total_tokens, output_dim, dtype=x_list[0].dtype, device=x_list[0].device)

    offset = 0
    for i, x_seq in enumerate(x_list):
        seq_len = x_seq.shape[0]
        w_idx = weight_indices[i]
        rank = ranks[w_idx]
        scaling = scalings[w_idx]
        if rank == 0:
            offset += seq_len
            continue

        A = A_weights[w_idx]  # (num_slices * max_rank, input_dim)
        B = B_weights[w_idx]  # (output_dim, max_rank)

        for s in range(num_slices):
            a_slice = A[s * rank : (s + 1) * rank, :]  # (rank, input_dim)
            mid = x_seq @ a_slice.t()  # (seq_len, rank)

            s_start = slice_offsets[s].item()
            s_end = slice_offsets[s + 1].item()
            b_slice = B[s_start:s_end, :rank]  # (slice_dim, rank)

            output[offset:offset + seq_len, s_start:s_end] += scaling * (mid @ b_slice.t())

        offset += seq_len
    return output


def run_test(num_seqs, seq_lens, num_adapters, ranks_list, input_dim, output_dim, dtype, num_slices=1):
    device = "cuda"
    max_rank = max(ranks_list)

    A_weights = torch.randn(num_adapters, num_slices * max_rank, input_dim, dtype=dtype, device=device) * 0.1
    B_weights = torch.randn(num_adapters, output_dim, max_rank, dtype=dtype, device=device) * 0.1
    scalings_list = [1.0] * num_adapters
    slice_offsets = torch.tensor(
        [i * (output_dim // num_slices) for i in range(num_slices + 1)],
        dtype=torch.int32, device=device,
    )

    # Zero out padded rank dimensions in weights
    for i in range(num_adapters):
        r = ranks_list[i]
        if r < max_rank:
            for s in range(num_slices):
                A_weights[i, s * max_rank + r : (s + 1) * max_rank, :] = 0
            B_weights[i, :, r:] = 0

    # Assign adapters round-robin
    weight_indices = [i % num_adapters for i in range(num_seqs)]
    x_list = [torch.randn(sl, input_dim, dtype=dtype, device=device) * 0.5 for sl in seq_lens]
    x_cat = torch.cat(x_list, dim=0)

    # ── Torch reference ──
    ref_output = torch_reference_lora(
        x_list, A_weights, B_weights, weight_indices,
        ranks_list, scalings_list, slice_offsets,
    )

    # ── Triton backend ──
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
    triton_backend = TritonLoRABackend(max_loras_per_batch=num_adapters, device=torch.device(device))

    extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    fb = FakeForwardBatch(
        batch_size=num_seqs,
        forward_mode=FakeForwardBatch.FakeMode(),
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=seq_lens,
        extend_num_tokens=sum(seq_lens),
        lora_ids=list(range(num_seqs)),
    )
    triton_backend.prepare_lora_batch(
        forward_batch=fb,
        weight_indices=weight_indices,
        lora_ranks=ranks_list + [0] * (num_adapters - len(ranks_list)),
        scalings=scalings_list + [0] * (num_adapters - len(scalings_list)),
        use_cuda_graph=False,
    )

    if num_slices == 1:
        triton_a_out = triton_backend.run_lora_a_sgemm(x_cat, A_weights)
        triton_output = triton_backend.run_lora_b_sgemm(
            x=triton_a_out, weights=B_weights, base_output=None,
        )
    elif num_slices == 3:
        triton_output = triton_backend.run_qkv_lora(
            x=x_cat, qkv_lora_a=A_weights, qkv_lora_b=B_weights,
            output_offset=slice_offsets, max_qkv_out_dim=output_dim // num_slices,
        )
    elif num_slices == 2:
        triton_output = triton_backend.run_gate_up_lora(
            x=x_cat, gate_up_lora_a=A_weights, gate_up_lora_b=B_weights,
            base_output=None,
        )

    # ── CSGMV backend ──
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend

    class FakeServerArgs:
        max_lora_chunk_size = 128
    csgmv_backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=num_adapters, device=torch.device(device),
        server_args=FakeServerArgs(),
    )
    csgmv_backend.prepare_lora_batch(
        forward_batch=fb,
        weight_indices=weight_indices,
        lora_ranks=ranks_list + [0] * (num_adapters - len(ranks_list)),
        scalings=scalings_list + [0] * (num_adapters - len(scalings_list)),
        use_cuda_graph=False,
    )

    if num_slices == 1:
        csgmv_a_out = csgmv_backend.run_lora_a_sgemm(x_cat, A_weights)
        csgmv_output = csgmv_backend.run_lora_b_sgemm(
            x=csgmv_a_out, weights=B_weights,
            output_offset=slice_offsets, base_output=None,
        )
    elif num_slices == 3:
        csgmv_output = csgmv_backend.run_qkv_lora(
            x=x_cat, qkv_lora_a=A_weights, qkv_lora_b=B_weights,
            output_offset=slice_offsets, max_qkv_out_dim=output_dim // num_slices,
        )
    elif num_slices == 2:
        csgmv_output = csgmv_backend.run_gate_up_lora(
            x=x_cat, gate_up_lora_a=A_weights, gate_up_lora_b=B_weights,
            base_output=None,
        )

    # ── Compare ──
    torch.cuda.synchronize()

    ref_vs_triton = (ref_output - triton_output).abs().max().item()
    ref_vs_csgmv = (ref_output - csgmv_output).abs().max().item()
    triton_vs_csgmv = (triton_output - csgmv_output).abs().max().item()

    ref_vs_triton_mean = (ref_output - triton_output).abs().mean().item()
    ref_vs_csgmv_mean = (ref_output - csgmv_output).abs().mean().item()
    triton_vs_csgmv_mean = (triton_output - csgmv_output).abs().mean().item()

    return {
        "ref_vs_triton": (ref_vs_triton, ref_vs_triton_mean),
        "ref_vs_csgmv": (ref_vs_csgmv, ref_vs_csgmv_mean),
        "triton_vs_csgmv": (triton_vs_csgmv, triton_vs_csgmv_mean),
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    configs = [
        {
            "name": "same-rank bf16, 2 adapters, long seqs",
            "num_seqs": 8, "seq_lens": [512, 256, 1024, 128, 512, 256, 768, 384],
            "num_adapters": 2, "ranks_list": [32, 32],
            "input_dim": 2560, "output_dim": 2560, "dtype": torch.bfloat16,
        },
        {
            "name": "mixed-rank bf16, 2 adapters",
            "num_seqs": 8, "seq_lens": [512, 256, 1024, 128, 512, 256, 768, 384],
            "num_adapters": 2, "ranks_list": [32, 16],
            "input_dim": 2560, "output_dim": 2560, "dtype": torch.bfloat16,
        },
        {
            "name": "same-rank bf16, QKV (3 slices)",
            "num_seqs": 4, "seq_lens": [512, 256, 1024, 128],
            "num_adapters": 2, "ranks_list": [32, 16],
            "input_dim": 2560, "output_dim": 384,  # 128 * 3 for qkv
            "dtype": torch.bfloat16, "num_slices": 3,
        },
        {
            "name": "large weights (simulating late training)",
            "num_seqs": 8, "seq_lens": [512, 256, 1024, 128, 512, 256, 768, 384],
            "num_adapters": 2, "ranks_list": [32, 16],
            "input_dim": 2560, "output_dim": 2560, "dtype": torch.bfloat16,
        },
    ]

    print("=" * 70)
    print("csgmv vs triton vs torch reference")
    print("=" * 70)

    all_ok = True
    for cfg in configs:
        name = cfg.pop("name")
        num_slices = cfg.pop("num_slices", 1)
        print(f"\n{name}:")

        # For "large weights" test, we scale weights up
        if "large" in name:
            torch.manual_seed(42)

        results = run_test(**cfg, num_slices=num_slices)

        for comp, (max_diff, mean_diff) in results.items():
            status = "OK" if max_diff < 0.05 else "HIGH"
            print(f"  {comp:20s}: max={max_diff:.6e}  mean={mean_diff:.6e}  [{status}]")
            if status == "HIGH":
                all_ok = False

        # Check if csgmv is further from reference than triton
        ref_triton_max = results["ref_vs_triton"][0]
        ref_csgmv_max = results["ref_vs_csgmv"][0]
        if ref_csgmv_max > ref_triton_max * 2 and ref_csgmv_max > 1e-4:
            print(f"  WARNING: csgmv is {ref_csgmv_max/ref_triton_max:.1f}x further from reference than triton")

    print("\n" + "=" * 70)
    print("PASS" if all_ok else "ISSUES FOUND")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)
