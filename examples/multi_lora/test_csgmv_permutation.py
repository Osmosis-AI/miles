"""Test: Does csgmv's permutation + segment construction correctly
cover every token exactly once?

If a token is missed or double-counted, the LoRA contribution for that
token is wrong (zero or duplicated), producing wrong logits.

Usage:
    python examples/multi_lora/test_csgmv_permutation.py
"""

import sys
import torch
from dataclasses import dataclass
from typing import Optional

from sglang.srt.model_executor.forward_batch_info import ForwardMode
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


def verify_permutation(name, num_seqs, seq_lens, weight_indices, num_adapters, ranks, mode=ForwardMode.EXTEND):
    """Verify that permutation + segments cover all tokens exactly once."""
    device = "cuda"
    total_tokens = sum(seq_lens)

    extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    fb = FakeForwardBatch(
        batch_size=num_seqs,
        forward_mode=mode,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=seq_lens,
        extend_num_tokens=total_tokens,
        lora_ids=list(range(num_seqs)),
    )

    backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=num_adapters,
        device=torch.device(device),
        server_args=FakeServerArgs(),
    )

    scalings = [1.0] * num_adapters
    backend.prepare_lora_batch(
        forward_batch=fb,
        weight_indices=weight_indices,
        lora_ranks=ranks,
        scalings=scalings,
        use_cuda_graph=False,
    )

    bi = backend.batch_info
    perm = bi.permutation[:total_tokens].cpu()
    seg_indptr = bi.seg_indptr[:bi.num_segments + 1].cpu()
    seg_wi = bi.weight_indices[:bi.num_segments].cpu()

    errors = []

    # 1. Permutation must be a valid permutation of [0, total_tokens)
    perm_sorted = perm.sort()[0]
    expected = torch.arange(total_tokens, dtype=perm.dtype)
    if not torch.equal(perm_sorted, expected):
        missing = set(range(total_tokens)) - set(perm.tolist())
        duped = [x for x in perm.tolist() if perm.tolist().count(x) > 1]
        errors.append(f"Permutation is not a valid permutation! missing={missing}, duped={set(duped)}")

    # 2. Segments must cover [0, total_tokens) contiguously
    if seg_indptr[0] != 0:
        errors.append(f"seg_indptr[0] = {seg_indptr[0]}, expected 0")
    if seg_indptr[bi.num_segments] != total_tokens:
        errors.append(f"seg_indptr[-1] = {seg_indptr[bi.num_segments]}, expected {total_tokens}")

    # 3. Each segment's tokens must belong to the correct adapter
    token_to_adapter = {}
    offset = 0
    for seq_idx, sl in enumerate(seq_lens):
        for t in range(sl):
            token_to_adapter[offset + t] = weight_indices[seq_idx]
        offset += sl

    for seg_idx in range(bi.num_segments):
        seg_start = seg_indptr[seg_idx].item()
        seg_end = seg_indptr[seg_idx + 1].item()
        adapter = seg_wi[seg_idx].item()

        for logical_pos in range(seg_start, seg_end):
            physical_pos = perm[logical_pos].item()
            actual_adapter = token_to_adapter[physical_pos]
            if actual_adapter != adapter:
                errors.append(
                    f"Seg {seg_idx} (adapter {adapter}): logical={logical_pos} "
                    f"physical={physical_pos} belongs to adapter {actual_adapter}"
                )
                if len(errors) > 10:
                    errors.append("... (truncated)")
                    break
        if len(errors) > 10:
            break

    # 4. Every token must appear in exactly one segment
    covered = set()
    for seg_idx in range(bi.num_segments):
        seg_start = seg_indptr[seg_idx].item()
        seg_end = seg_indptr[seg_idx + 1].item()
        for logical_pos in range(seg_start, seg_end):
            physical_pos = perm[logical_pos].item()
            if physical_pos in covered:
                errors.append(f"Token {physical_pos} covered by multiple segments")
            covered.add(physical_pos)

    uncovered = set(range(total_tokens)) - covered
    if uncovered:
        errors.append(f"Tokens not covered by any segment: {uncovered}")

    if errors:
        print(f"  [{name}] FAIL:")
        for e in errors:
            print(f"    {e}")
        return False
    else:
        print(f"  [{name}] PASS (tokens={total_tokens}, segments={bi.num_segments}, chunk={bi.max_len})")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("csgmv permutation + segment correctness tests")
    print("=" * 60)

    results = []

    # Basic extend
    results.append(verify_permutation(
        "basic extend", num_seqs=4, seq_lens=[100, 200, 150, 50],
        weight_indices=[0, 1, 0, 1], num_adapters=2, ranks=[32, 16],
    ))

    # Single adapter
    results.append(verify_permutation(
        "single adapter", num_seqs=4, seq_lens=[100, 200, 150, 50],
        weight_indices=[0, 0, 0, 0], num_adapters=2, ranks=[32, 16],
    ))

    # Many adapters
    results.append(verify_permutation(
        "4 adapters", num_seqs=8,
        seq_lens=[100, 200, 150, 50, 300, 75, 125, 250],
        weight_indices=[0, 1, 2, 3, 0, 1, 2, 3], num_adapters=4,
        ranks=[32, 16, 32, 16],
    ))

    # Skewed: 1 long + many short
    results.append(verify_permutation(
        "skewed", num_seqs=8,
        seq_lens=[2048, 64, 64, 64, 64, 64, 64, 64],
        weight_indices=[0, 1, 0, 1, 0, 1, 0, 1], num_adapters=2,
        ranks=[32, 16],
    ))

    # Decode (1 token per seq)
    results.append(verify_permutation(
        "decode", num_seqs=32, seq_lens=[1] * 32,
        weight_indices=[i % 2 for i in range(32)], num_adapters=2,
        ranks=[32, 16], mode=ForwardMode.DECODE,
    ))

    # Large batch
    results.append(verify_permutation(
        "large batch", num_seqs=64,
        seq_lens=[128 + (i * 17) % 256 for i in range(64)],
        weight_indices=[i % 2 for i in range(64)], num_adapters=2,
        ranks=[32, 16],
    ))

    # Edge: all same length
    results.append(verify_permutation(
        "same length", num_seqs=8, seq_lens=[256] * 8,
        weight_indices=[0, 1, 0, 1, 0, 1, 0, 1], num_adapters=2,
        ranks=[32, 16],
    ))

    # Edge: length 1 extend
    results.append(verify_permutation(
        "length-1 extend", num_seqs=4, seq_lens=[1, 1, 1, 1],
        weight_indices=[0, 1, 0, 1], num_adapters=2,
        ranks=[32, 16],
    ))

    # Realistic: dapo (long) + gsm8k (short), mixed
    results.append(verify_permutation(
        "dapo+gsm8k realistic", num_seqs=16,
        seq_lens=[800, 120, 950, 80, 700, 150, 1100, 90,
                  650, 130, 880, 70, 750, 110, 1050, 100],
        weight_indices=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        num_adapters=2, ranks=[32, 16],
    ))

    print("\n" + "=" * 60)
    all_pass = all(results)
    print("ALL PASS" if all_pass else "FAILURES DETECTED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
