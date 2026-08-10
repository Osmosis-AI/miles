"""Prove a LoRA adapter checkpoint round-trips exactly across TP/PP/EP shards.

Fills every adapter parameter with rank-distinct values, saves the checkpoint, zeroes
the parameters, reloads, and compares bitwise. The values are deliberately a function of
the global rank: with EP > 1 each rank owns different routed experts, so a shard name
that omits the EP rank makes the ranks overwrite each other's file and reload a
neighbour's experts. That failure is invisible to a name-based check -- the parameter
names are identical on every EP rank -- and is exactly what this compares values to catch.

Run under torchrun with the training spec's parallel + LoRA args, e.g. by replaying the
argv a run was launched with:

    ROUNDTRIP_DIR=/tmp/rt torchrun --nproc-per-node 8 tools/verify_lora_roundtrip.py <argv...>

Exits non-zero on any mismatch.
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.lora_utils import (
    _is_adapter_param_name,
    load_lora_adapter,
    save_lora_checkpoint,
)
from miles.backends.megatron_utils.model import initialize_model_and_optimizer
from miles.utils.arguments import parse_args
from miles.utils.distributed_utils import init_gloo_group


def adapter_params(model):
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if _is_adapter_param_name(name):
                yield name, param


def fill_rank_distinct(model, rank: int) -> dict[str, torch.Tensor]:
    """Give every adapter tensor values no other rank would produce, and snapshot them."""
    generator = torch.Generator(device="cpu").manual_seed(1000 + rank)
    snapshot = {}
    for name, param in adapter_params(model):
        values = torch.randn(param.shape, generator=generator, dtype=torch.float32)
        param.data.copy_(values.to(device=param.device, dtype=param.dtype))
        snapshot[name] = param.data.clone().cpu()
    return snapshot


def main(args) -> int:
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl")
    # The adapter saver elects one shard writer per (tp, pp, ep) coordinate over the
    # gloo group; the Ray train actor creates it during setup, a standalone torchrun
    # does not. Idempotent, and collective -- every rank reaches it.
    init_gloo_group()
    init(args)
    args.no_load_optim = True
    args.no_load_rng = True
    model, _, _, _ = initialize_model_and_optimizer(args)

    rank = dist.get_rank()
    out_dir = Path(os.environ.get("ROUNDTRIP_DIR", "/tmp/lora_roundtrip"))

    snapshot = fill_rank_distinct(model, rank)
    save_lora_checkpoint(model, args, str(out_dir))

    for _, param in adapter_params(model):
        param.data.zero_()
    loaded, _ = load_lora_adapter(model, str(out_dir))
    if not loaded:
        print(f"[rank {rank}] FAIL: no adapter checkpoint found under {out_dir}", flush=True)
        return 1

    mismatched = [
        name for name, param in adapter_params(model) if not torch.equal(param.data.cpu(), snapshot[name])
    ]
    ok = not mismatched
    print(
        f"[rank {rank}] {'OK' if ok else 'FAIL'}: {len(snapshot)} adapter tensors, "
        f"{len(mismatched)} mismatched{'' if ok else f' e.g. {mismatched[:2]}'}",
        flush=True,
    )

    verdicts = [None] * dist.get_world_size()
    dist.all_gather_object(verdicts, ok)
    if rank == 0:
        shards = sorted(p.name for p in out_dir.glob("adapter_megatron_*.pt"))
        print(f"\nshards written: {len(shards)}\n  " + "\n  ".join(shards), flush=True)
        print(f"\n{'PASS' if all(verdicts) else 'FAIL'}: {sum(verdicts)}/{len(verdicts)} ranks round-tripped exactly")
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main(parse_args()))
