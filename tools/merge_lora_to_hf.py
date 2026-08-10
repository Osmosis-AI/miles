"""Fold a trained LoRA adapter into the base model and write a standalone HF checkpoint.

Training saves adapter-only checkpoints (``{save}/iter_*/adapter``); a merged model is
only written when ``--save-hf`` is set, which a training run that did not set it never
produces. This rebuilds the merged model after the fact.

It reuses the training code path rather than merging tensors by hand: the model is
constructed with the *same* LoRA and parallel arguments the run used, the adapter is
restored through the normal checkpoint path, and ``save_hf_model`` folds the adapter
into the base weights. Hand-merging is a trap here -- with ``--experts-shared-outer-loras``
the expert LoRAs carry a broadcast axis against 3D stacked expert weights, and a wrong
transpose yields a model that still emits fluent text.

Run under torchrun with the training spec's parallel + LoRA args:

    PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/merge_lora_to_hf.py \\
        --hf-checkpoint Qwen/Qwen3.6-35B-A3B \\
        --lora-adapter-path /weka/.../iter_0000196/adapter \\
        --save-hf /weka/.../merged \\
        --lora-rank 32 --lora-alpha 32 --experts-shared-outer-loras \\
        --target-modules '...' \\
        --tensor-model-parallel-size 2 --expert-model-parallel-size 8 \\
        --expert-tensor-parallel-size 1

Writes the merged model to ``--save-hf`` and a re-exported adapter to ``{save-hf}/adapter``.
"""

import json
import os
from pathlib import Path

import torch

# Import from the defining modules, not the package: megatron_utils/__init__.py
# re-exports neither of these. tools/convert_to_hf.py still reaches for
# `megatron_utils.init` and is dead code against this version.
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model import initialize_model_and_optimizer, save_hf_model
from miles.utils.arguments import parse_args

# The only weights the base checkpoint carries that a miles model legitimately does not
# export: miles builds with mtp_num_layers=0, so the multi-token-prediction head never
# exists to be merged. Nothing served here uses it.
UNEXPORTED_PREFIX = "mtp."


def _shard_tensor_bytes(shard: Path) -> dict[str, int]:
    """Tensor name -> byte length, read from the safetensors header without loading data."""
    with open(shard, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return {
        name: entry["data_offsets"][1] - entry["data_offsets"][0]
        for name, entry in header.items()
        if name != "__metadata__"
    }


def _verify_and_repair_index(path: Path) -> None:
    """Drop index entries for tensors that were never written, and re-total.

    ``strict=False`` gets the incomplete shards written, but the index the bridge then
    emits still lists every key those shards were *supposed* to hold, and copies the
    source checkpoint's ``total_size``. So the index has to be reconciled against what
    is actually on disk -- and anything missing beyond the MTP head means the export
    silently lost real weights, which is a hard failure rather than a warning.
    """
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        raise RuntimeError(f"no {index_path.name} under {path}: the merged save did not run")
    index = json.loads(index_path.read_text())

    present: dict[str, int] = {}
    for shard in sorted(set(index["weight_map"].values())):
        if (path / shard).exists():
            present.update(_shard_tensor_bytes(path / shard))

    absent = sorted(set(index["weight_map"]) - present.keys())
    lost = [name for name in absent if not name.startswith(UNEXPORTED_PREFIX)]
    if lost:
        raise RuntimeError(f"merged model is missing {len(lost)} tensors, e.g. {lost[:5]}")

    index["weight_map"] = {name: shard for name, shard in index["weight_map"].items() if name in present}
    index["metadata"]["total_size"] = sum(present.values())
    index_path.write_text(json.dumps(index, indent=2))
    print(f"index: {len(present)} tensors written, {len(absent)} {UNEXPORTED_PREFIX}* entries dropped")


def main(args):
    # miles' init() configures model parallel but does not create the process group
    # itself -- despite _initialize_distributed's docstring. In the normal flow the
    # Ray actor has already done it; under torchrun nothing has, and
    # mpu.initialize_model_parallel trips `assert torch.distributed.is_initialized()`.
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        torch.distributed.init_process_group(backend="nccl")

    init(args)

    # Nothing here trains; the optimizer and RNG state in the checkpoint are irrelevant
    # and the adapter-only checkpoints do not carry them.
    args.no_load_optim = True
    args.no_load_rng = True

    model, _, _, _ = initialize_model_and_optimizer(args)

    # Collective: every rank must call it. strict=False because the MTP head is absent
    # by construction and shares its shards with layers 38-39, the final norm and
    # lm_head -- under the default those four would be discarded along with it, and the
    # writer reports success either way.
    save_hf_model(args, 0, model, strict=False)

    # Rank 0 is the writer, and the index it leaves behind describes the source
    # checkpoint rather than what it wrote.
    if torch.distributed.get_rank() == 0:
        _verify_and_repair_index(Path(args.save_hf.format(rollout_id=0)))


if __name__ == "__main__":
    main(parse_args())
