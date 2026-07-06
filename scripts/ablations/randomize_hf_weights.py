"""Rewrite an HF safetensors checkpoint with seeded random weights.

Shard-rewrite: iterates model.safetensors.index.json without instantiating the
model, so it works for any architecture and needs only shard-sized memory.
Per-tensor seeding is order-independent (seed ^ crc32(name)), so the output is
reproducible regardless of shard iteration order.

Init rules (by tensor name, first match wins):
  *norm*.weight            -> ones   (RMSNorm/LayerNorm/q_norm/k_norm)
  *.bias                   -> zeros
  *A_log*                  -> log(uniform(1, 16))   (GDN decay, stable range)
  *dt_bias*                -> normal(0, 0.02)
  everything else          -> normal(0, 0.02)       (embeddings, projections,
                                                     router gate, experts, conv)
"""

import argparse
import json
import shutil
import zlib
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

INIT_STD = 0.02


def randomize_tensor(name: str, tensor: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed ^ zlib.crc32(name.encode()))
    shape = tensor.shape
    if "norm" in name and name.endswith(".weight"):
        out = torch.ones(shape, dtype=torch.float32)
    elif name.endswith(".bias"):
        out = torch.zeros(shape, dtype=torch.float32)
    elif "A_log" in name:
        out = torch.empty(shape, dtype=torch.float32).uniform_(1.0, 16.0, generator=gen).log()
    else:
        out = torch.empty(shape, dtype=torch.float32).normal_(0.0, INIT_STD, generator=gen)
    assert torch.isfinite(out).all(), f"non-finite init for {name}"
    return out.to(tensor.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="HF checkpoint dir to randomize")
    parser.add_argument("--dst", type=Path, required=True, help="output dir")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    done_marker = args.dst / ".done"
    if done_marker.exists():
        print(f"[cache-hit] {args.dst}")
        return

    args.dst.mkdir(parents=True, exist_ok=True)

    index_path = args.src / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    for shard in shards:
        tensors = {}
        with safe_open(args.src / shard, framework="pt") as f:
            for name in f.keys():
                tensors[name] = randomize_tensor(name, f.get_tensor(name), args.seed)
        save_file(tensors, args.dst / shard, metadata={"format": "pt"})
        print(f"[randomized] {shard} ({len(tensors)} tensors)")

    for entry in args.src.iterdir():
        if entry.name.endswith(".safetensors") or entry.name == ".done":
            continue
        if entry.name == "config.json":
            config = json.loads(entry.read_text())
            config.pop("quantization_config", None)
            (args.dst / entry.name).write_text(json.dumps(config, indent=2))
        elif entry.is_file():
            shutil.copy2(entry, args.dst / entry.name)

    done_marker.touch()
    print(f"[done] {args.dst} (seed {args.seed})")


if __name__ == "__main__":
    main()
