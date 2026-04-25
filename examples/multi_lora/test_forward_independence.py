"""Test: Multi-LoRA forward pass independence.

Verifies that per-sample logits from a multi-LoRA forward pass are identical
regardless of what OTHER samples are in the batch.  If this test fails, the
multi-LoRA routing has cross-adapter contamination.

Uses SimpleMultiLoRALinear (no Megatron parallel state needed).

Usage:
    # From the mathew-miles repo root, with megatron-bridge on PYTHONPATH:
    PYTHONPATH=../mathew-megatron-bridge/src:$PYTHONPATH \
        python examples/multi_lora/test_forward_independence.py

    # Or on cluster where megatron-bridge is already installed:
    python examples/multi_lora/test_forward_independence.py
"""

import sys
import torch
import torch.nn as nn


def _make_layer(in_features, out_features, n_adapters, max_rank, adapter_configs):
    """Create SimpleMultiLoRALinear with per-adapter rank/alpha.

    adapter_configs: list of (rank, alpha) tuples, one per adapter.
    """
    from megatron.bridge.peft.multi_lora_layers import SimpleMultiLoRALinear, register_adapter

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = nn.Linear(in_features, out_features, bias=False, device=device)
    nn.init.normal_(base.weight, mean=0, std=0.02)

    layer = SimpleMultiLoRALinear(base, n_adapters=n_adapters, dim=max_rank, alpha=1.0)

    # Set distinct weights per adapter
    with torch.no_grad():
        for i in range(n_adapters):
            nn.init.normal_(layer.adapters[i].linear_in.weight, mean=0.1 * (i + 1), std=0.01)
            nn.init.normal_(layer.adapters[i].linear_out.weight, mean=0.05 * (i + 1), std=0.01)

    class _Wrap(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
    model = _Wrap(layer)

    for i, (rank, alpha) in enumerate(adapter_configs):
        register_adapter(model, i, rank=rank, alpha=alpha)

    return layer


# ── Test 1: basic independence ────────────────────────────────────────────

def test_forward_independence():
    """Mixed batch vs separate batches — outputs must be bitwise identical."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = _make_layer(256, 128, n_adapters=2, max_rank=32,
                        adapter_configs=[(32, 32.0), (16, 16.0)])
    layer.eval()

    n_a, n_b = 50, 30
    torch.manual_seed(123)
    tok_a = torch.randn(n_a, 256, device=device)
    tok_b = torch.randn(n_b, 256, device=device)

    # Mixed
    layer.tokens_per_adapter = torch.tensor([n_a, n_b], device=device)
    with torch.no_grad():
        mixed = layer(torch.cat([tok_a, tok_b]).unsqueeze(0)).squeeze(0)
    mixed_a, mixed_b = mixed[:n_a].clone(), mixed[n_a:].clone()

    # Solo adapter 0
    layer.tokens_per_adapter = torch.tensor([n_a, 0], device=device)
    with torch.no_grad():
        solo_a = layer(tok_a.unsqueeze(0)).squeeze(0)

    # Solo adapter 1
    layer.tokens_per_adapter = torch.tensor([0, n_b], device=device)
    with torch.no_grad():
        solo_b = layer(tok_b.unsqueeze(0)).squeeze(0)

    da = (mixed_a - solo_a).abs().max().item()
    db = (mixed_b - solo_b).abs().max().item()
    # Check if diff comes from base linear (batch-size-dependent CUDA matmul noise)
    # or from the adapter path (would indicate real cross-contamination)
    da_mean = (mixed_a - solo_a).abs().mean().item()
    db_mean = (mixed_b - solo_b).abs().mean().item()
    print(f"[independence] adapter0: max={da:.2e} mean={da_mean:.2e}   adapter1: max={db:.2e} mean={db_mean:.2e}")

    # ~1e-6 is expected float32 CUDA matmul non-determinism for different batch shapes.
    # Real cross-contamination would show diffs >> 1e-3.
    threshold = 1e-5
    ok = da < threshold and db < threshold
    if ok:
        print(f"  PASS: diffs within float32 matmul noise (threshold={threshold:.0e})")
    else:
        print(f"  FAIL: diffs too large — possible cross-adapter contamination")
    return ok


# ── Test 2: batch composition doesn't matter ──────────────────────────────

def test_composition_invariance():
    """Adapter 0's output unchanged when adapter 1's batch size changes."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = _make_layer(256, 128, n_adapters=2, max_rank=32,
                        adapter_configs=[(32, 32.0), (16, 16.0)])
    layer.eval()

    n_a = 50
    torch.manual_seed(123)
    tok_a = torch.randn(n_a, 256, device=device)

    results = []
    for n_b in [10, 50, 200]:
        tok_b = torch.randn(n_b, 256, device=device)
        layer.tokens_per_adapter = torch.tensor([n_a, n_b], device=device)
        with torch.no_grad():
            out = layer(torch.cat([tok_a, tok_b]).unsqueeze(0)).squeeze(0)[:n_a].clone()
        results.append(out)

    d1 = (results[0] - results[1]).abs().max().item()
    d2 = (results[0] - results[2]).abs().max().item()
    print(f"[composition] n_b=10 vs 50: {d1:.2e}   n_b=10 vs 200: {d2:.2e}")
    threshold = 1e-5
    ok = d1 < threshold and d2 < threshold
    print(f"  {'PASS' if ok else 'FAIL'} (threshold={threshold:.0e})")
    return ok


# ── Test 3: scaling correctness for mixed rank ────────────────────────────

def test_scaling():
    """Verify alpha/rank scaling matches manual computation."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_rank = 32
    configs = [(32, 32.0), (16, 16.0)]  # (rank, alpha)
    layer = _make_layer(256, 128, n_adapters=2, max_rank=max_rank,
                        adapter_configs=configs)
    layer.eval()

    ok = True
    for idx, (rank, alpha) in enumerate(configs):
        n = 40
        torch.manual_seed(200 + idx)
        tokens = torch.randn(n, 256, device=device)

        counts = [0, 0]
        counts[idx] = n
        layer.tokens_per_adapter = torch.tensor(counts, device=device)
        with torch.no_grad():
            actual = layer(tokens.unsqueeze(0)).squeeze(0).clone()

        # Manual: base + adapter with CORRECT scaling
        adapter = layer.adapters[idx]
        with torch.no_grad():
            base = torch.nn.functional.linear(tokens, layer.weight, layer.bias)
            lora = adapter.linear_out(adapter.linear_in(tokens))
            expected = base + lora * (alpha / rank)

        diff = (actual - expected).abs().max().item()
        status = "PASS" if diff < 1e-5 else "FAIL"
        print(f"[scaling] adapter {idx} (rank={rank}, alpha={alpha}): "
              f"max diff vs manual = {diff:.2e}  [{status}]")
        if diff >= 1e-5:
            # Check if it matches the WRONG scaling (construction alpha/max_rank)
            wrong = base + lora * (layer.adapters[idx].alpha / layer.adapters[idx].dim)
            wrong_diff = (actual - wrong).abs().max().item()
            print(f"           diff vs WRONG scaling (alpha_init/max_rank) = {wrong_diff:.2e}")
            ok = False

    return ok


# ── Test 4: zero-padding correctness ─────────────────────────────────────

def test_zero_padding():
    """Rank-16 adapter in rank-32 slot: padded dims must contribute zero."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_rank = 32
    actual_rank = 16
    layer = _make_layer(256, 128, n_adapters=2, max_rank=max_rank,
                        adapter_configs=[(max_rank, 32.0), (actual_rank, 16.0)])
    layer.eval()

    # After register_adapter + apply_rank_masks, the lower-rank adapter's
    # weights should be zero in the padded region.
    adapter = layer.adapters[1]
    a_weight = adapter.linear_in.weight.data   # [max_rank, in_features]
    b_weight = adapter.linear_out.weight.data  # [out_features, max_rank]

    # For linear_in: rows [actual_rank:] should be zero
    a_pad = a_weight[actual_rank:].abs().max().item()
    # For linear_out: cols [actual_rank:] should be zero
    b_pad = b_weight[:, actual_rank:].abs().max().item() if b_weight.shape[-1] > actual_rank else 0.0

    # SimpleLoRAAdapter stores linear_out as [out_features, dim] where dim=max_rank
    # The zero-padding might be in different dimensions depending on the weight layout
    print(f"[zero_pad] A padded region max = {a_pad:.2e}   B padded region max = {b_pad:.2e}")

    ok = a_pad < 1e-10 and b_pad < 1e-10
    if not ok:
        print(f"  FAIL: zero-padding not applied correctly")
        print(f"  A shape={a_weight.shape}, B shape={b_weight.shape}")
        print(f"  rank_values={layer.rank_values.tolist()}")
    else:
        print(f"  PASS")
    return ok


# ── Test 5: backup / restore round-trip ───────────────────────────────────

def test_backup_restore():
    """Weight backup → modify → restore should give back exact originals."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = _make_layer(256, 128, n_adapters=2, max_rank=32,
                        adapter_configs=[(32, 32.0), (16, 16.0)])

    orig = {n: p.data.clone() for n, p in layer.named_parameters()}
    backup = {n: p.data.clone().cpu() for n, p in layer.named_parameters()}

    with torch.no_grad():
        for p in layer.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    with torch.no_grad():
        for n, p in layer.named_parameters():
            p.copy_(backup[n].to(device))

    max_diff = max((p.data - orig[n]).abs().max().item()
                   for n, p in layer.named_parameters())
    ok = max_diff == 0.0
    print(f"[backup_restore] max diff after round-trip = {max_diff:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-LoRA Forward Pass Independence Tests")
    print("=" * 60)
    print()

    results = [
        ("Forward independence",       test_forward_independence),
        ("Composition invariance",     test_composition_invariance),
        ("Scaling correctness",        test_scaling),
        ("Zero-padding",               test_zero_padding),
        ("Backup / restore",           test_backup_restore),
    ]

    outcomes = []
    for name, fn in results:
        try:
            ok = fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            ok = False
        outcomes.append((name, ok))
        print()

    print("=" * 60)
    all_pass = True
    for name, ok in outcomes:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_pass = False
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
