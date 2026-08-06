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

import miles.backends.megatron_utils as megatron_utils
from miles.backends.megatron_utils.model import save_hf_model
from miles.utils.arguments import parse_args


def main(args):
    from miles.utils.ft_utils.indep_dp import IndepDPInfo

    megatron_utils.init(
        args,
        indep_dp_store_addr=None,
        indep_dp_info=IndepDPInfo.create_trivial(),
    )

    # Nothing here trains; the optimizer and RNG state in the checkpoint are irrelevant
    # and the adapter-only checkpoints do not carry them.
    args.no_load_optim = True
    args.no_load_rng = True

    model, _, _, _ = megatron_utils.initialize_model_and_optimizer(args)

    # Collective: every rank must call it.
    save_hf_model(args, 0, model)


if __name__ == "__main__":
    main(parse_args())
