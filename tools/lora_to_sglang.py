"""Rebuild a servable LoRA adapter from a miles LoRA checkpoint.

``save_lora_checkpoint`` writes two things: an HF adapter (``adapter_model.bin``) via
the bridge, and Megatron-native per-rank shards (``adapter_megatron_tp*_pp*.pt``). The
HF adapter is faithful for every module except the routed MoE experts, where the bridge
collapses the per-expert axis -- ``experts.gate_up_proj.lora_B`` comes out as a single
``(out, r)`` matrix where the model has one per expert. This tool takes the HF adapter
for everything else and rebuilds those two tensors from the native shards, which do
carry the expert axis.

SGLang accepts the result: its loader handles routed-expert LoRA either per-expert
(``...experts.0.<module>...``) or shared-outer (``...experts.<module>...``, a 3D tensor
with the expert dim in the shape) -- see ``srt/lora/lora.py``. This writes the latter,
which is the layout ``--experts-shared-outer-loras`` trains.

    python3 tools/lora_to_sglang.py \\
        --adapter-dir /weka/.../iter_0000196/adapter \\
        --out /weka/.../adapter_sglang \\
        --num-experts 256 \\
        --verify-against /weka/.../merged --base /osmosis/models/Qwen/Qwen3.6-35B-A3B

WARNING on this checkpoint family: the native shards are named by ``(tp, pp)`` only and
are written from ``dp_cp rank 0``, so with EP > 1 they hold one expert-parallel shard,
not all of them. The rebuilt tensor therefore repeats that shard across the expert axis
-- which is exactly what ``load_lora_adapter`` already does when it restores, so the
result matches a merged model built from the same checkpoint. It is *not* the adapter
that was trained. The tool reports how many distinct experts it actually had.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

# Megatron adapter leaf -> (HF module, HF lora side)
_EXPERT_PARTS = {
    ("linear_fc1", "linear_in"): ("gate_up_proj", "lora_A"),
    ("linear_fc1", "linear_out"): ("gate_up_proj", "lora_B"),
    ("linear_fc2", "linear_in"): ("down_proj", "lora_A"),
    ("linear_fc2", "linear_out"): ("down_proj", "lora_B"),
}
_NATIVE_EXPERT = re.compile(
    r"language_model\.decoder\.layers\.(\d+)\.mlp\.experts\.(linear_fc[12])\.adapter\.(linear_in|linear_out)\.weight$"
)
_HF_EXPERT = re.compile(r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)\.(lora_A|lora_B)\.weight$")


def load_native_experts(adapter_dir: Path, shard: str) -> dict[tuple[int, str, str], torch.Tensor]:
    """(layer, hf_module, lora_side) -> native tensor, from one Megatron shard."""
    state = torch.load(adapter_dir / shard, map_location="cpu", weights_only=True)
    out = {}
    for name, tensor in state.items():
        m = _NATIVE_EXPERT.search(name)
        if m:
            layer, fc, side = int(m.group(1)), m.group(2), m.group(3)
            module, lora_side = _EXPERT_PARTS[(fc, side)]
            out[(layer, module, lora_side)] = tensor
    return out


def rebuild_expert_tensor(native: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, int]:
    """Expand a native expert tensor to the full expert axis. Returns (tensor, n_distinct).

    Shared-outer halves come out of Megatron 2D (one matrix for every expert) and are
    emitted with a leading axis of 1, which is what the bridge already does and what
    SGLang broadcasts. Per-expert halves are 3D ``(local_experts, ...)`` and get tiled
    up to ``num_experts``.
    """
    if native.dim() == 2:
        return native.unsqueeze(0).contiguous(), 1
    local = native.shape[0]
    if num_experts % local:
        raise ValueError(f"{num_experts} experts is not a multiple of the {local} in the shard")
    return native.repeat(num_experts // local, 1, 1).contiguous(), local


def convert(adapter_dir: Path, out_dir: Path, num_experts: int, shard: str) -> dict[str, torch.Tensor]:
    weights = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu", weights_only=True)
    native = load_native_experts(adapter_dir, shard)

    rebuilt, distinct = 0, set()
    for name in list(weights):
        m = _HF_EXPERT.search(name)
        if not m:
            continue
        key = (int(m.group(1)), m.group(2), m.group(3))
        if key not in native:
            raise KeyError(f"{name} has no counterpart in {shard}")
        tensor, n = rebuild_expert_tensor(native[key], num_experts)
        weights[name] = tensor
        distinct.add(n)
        rebuilt += 1

    # The fused modules (q/k/v, in_proj_*, shared_expert gate/up) share one lora_A
    # storage, which safetensors refuses to serialize; clone so each key owns its data.
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().clone().contiguous() for k, v in weights.items()}, out_dir / "adapter_model.safetensors")
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))

    per_expert = sorted(n for n in distinct if n > 1)
    print(f"wrote {len(weights)} tensors to {out_dir}, {rebuilt} expert tensors rebuilt")
    print(f"distinct experts available in {shard}: {per_expert or ['n/a']} of {num_experts}")
    if per_expert and per_expert[0] < num_experts:
        print(f"WARNING: expert LoRA repeats every {per_expert[0]} experts -- the shard holds one EP slice")
    return weights


def verify(weights: dict[str, torch.Tensor], merged: Path, base: Path, samples: int) -> None:
    """Check reconstructed ``B @ A`` against ``merged - base`` for a sample of modules."""
    from safetensors import safe_open

    def slice_of(root: Path, name: str):
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        return safe_open(root / index[name], framework="pt").get_slice(name)

    def base_param_name(lora_name: str, index: dict) -> str | None:
        """Stacked MoE expert params are stored without the trailing ``.weight``."""
        stem = lora_name.replace(".lora_A.weight", "")
        return next((c for c in (f"{stem}.weight", stem) if c in index), None)

    def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.flatten(), y.flatten()
        return float(torch.dot(x, y) / (x.norm() * y.norm()))

    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    pairs = [n for n in weights if n.endswith("lora_A.weight")][:samples]
    print(f"\nverifying {len(pairs)} modules against {merged.name} - {base.name}")
    for a_name in pairs:
        b_name = a_name.replace("lora_A", "lora_B")
        hf_name = base_param_name(a_name, index)
        if hf_name is None:
            print(f"  skip {a_name} (no base-model counterpart)")
            continue
        a, b = weights[a_name].detach().float(), weights[b_name].detach().float()
        sb, sm = slice_of(base, hf_name), slice_of(merged, hf_name)
        # Expert tensors carry a leading expert axis on one or both sides; check expert 0.
        expert = a.dim() == 3 or b.dim() == 3
        if expert:
            delta = (sm[0:1].float() - sb[0:1].float()).squeeze(0)
            predicted = (b[0] if b.dim() == 3 else b).mm(a[0] if a.dim() == 3 else a)
        else:
            delta = sm[:].float() - sb[:].float()
            predicted = b.mm(a)
        print(f"  {hf_name:70s} cos {cosine(delta, predicted):+.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--shard", default="adapter_megatron_tp0_pp0.pt", help="native shard to take expert weights from")
    p.add_argument("--verify-against", type=Path, help="merged HF model to check the reconstruction against")
    p.add_argument("--base", type=Path, help="base HF checkpoint, required with --verify-against")
    p.add_argument("--verify-samples", type=int, default=6)
    args = p.parse_args()

    weights = convert(args.adapter_dir, args.out, args.num_experts, args.shard)
    if args.verify_against:
        verify(weights, args.verify_against, args.base, args.verify_samples)


if __name__ == "__main__":
    main()
