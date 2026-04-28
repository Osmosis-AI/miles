import dataclasses

import torch

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name
from miles.utils import megatron_bridge_utils
from miles.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        import miles_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type: str = "base"):
        # TODO: support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            if weight_type == "lora":
                named_weights = self._bridge.export_adapter_weights(
                    self.model,
                    cpu=False,
                    show_progress=False,
                )
            elif weight_type == "base":
                conversion_tasks = self._bridge.get_conversion_tasks(self.model)
                conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
                named_weights = self._bridge.export_hf_weights(
                    self.model,
                    cpu=False,
                    conversion_tasks=conversion_tasks,
                )

            # TODO: verify if postprocess_hf_param is needed for LoRA weights
            named_weights = (
                (
                    hf_param_name,
                    _postprocess_bridge_lora_param(
                        args=self.args,
                        hf_param_name=hf_param_name,
                        param=postprocess_hf_param(
                            args=self.args,
                            megatron_param_name=megatron_param_name,
                            hf_param_name=hf_param_name,
                            param=weight,
                        ),
                    )
                    if weight_type == "lora"
                    else postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    ),
                )
                for hf_param_name, weight, megatron_param_name in named_weights
            )

            if weight_type == "base":
                named_weights = ((n, t) for n, t in named_weights if not is_lora_weight_name(n))
            elif weight_type == "lora":
                named_weights = (
                    (n, t)
                    for n, t in named_weights
                    if is_lora_weight_name(n) and _is_rollout_lora_weight_name(n)
                )

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


def _postprocess_bridge_lora_param(args, hf_param_name, param):
    """Normalize Bridge adapter tensors to the PEFT layout expected by SGLang."""
    if ".lora_B." not in hf_param_name or param.ndim < 2:
        return param

    lora_rank = getattr(args, "lora_rank", None)
    if lora_rank is None:
        return param

    # Bridge can export LoRA B as rank-first ([r, out] or [..., r, out]).
    # SGLang's tensor loader expects PEFT layout ([out, r]) before TP slicing.
    if param.shape[-2] == lora_rank and param.shape[-1] != lora_rank:
        param = param.transpose(-1, -2).contiguous()

    if (
        getattr(args, "attention_output_gate", False)
        and (".qkv_proj.lora_B." in hf_param_name or ".q_proj.lora_B." in hf_param_name)
        and param.ndim == 2
    ):
        return _drop_qwen3_5_q_gate_rows(args, hf_param_name, param)

    return param


def _drop_qwen3_5_q_gate_rows(args, hf_param_name, param):
    """Convert Qwen3.5 gated QGKV LoRA B rows to SGLang's QKV-only layout."""
    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
        value_num_per_group = args.num_attention_heads // args.num_query_groups
    except AttributeError:
        return param

    if ".q_proj.lora_B." in hf_param_name:
        rows_per_group = 2 * value_num_per_group * head_dim
    else:
        rows_per_group = (2 * value_num_per_group + 2) * head_dim
    if param.shape[0] % rows_per_group != 0:
        raise ValueError(
            f"Cannot convert gated QKV LoRA tensor {hf_param_name}: "
            f"shape={tuple(param.shape)} is not divisible by rows_per_group={rows_per_group}"
        )

    num_groups = param.shape[0] // rows_per_group
    if ".q_proj.lora_B." in hf_param_name:
        q_with_gate = param.view(num_groups, 2 * value_num_per_group, head_dim, param.shape[1])
        return q_with_gate.view(num_groups, 2, value_num_per_group, head_dim, param.shape[1])[:, 0].reshape(
            -1, param.shape[1]
        ).contiguous()

    qgkv = param.view(num_groups, 2 * value_num_per_group + 2, head_dim, param.shape[1])
    q_with_gate, k, v = qgkv.split([2 * value_num_per_group, 1, 1], dim=1)
    q = q_with_gate.view(num_groups, 2, value_num_per_group, head_dim, param.shape[1])[:, 0]

    return torch.cat(
        [
            q.reshape(-1, param.shape[1]),
            k.reshape(-1, param.shape[1]),
            v.reshape(-1, param.shape[1]),
        ],
        dim=0,
    ).contiguous()


def _is_rollout_lora_weight_name(name: str) -> bool:
    """Keep adapter tensors that correspond to the language rollout model."""
    excluded_fragments = (
        "vision_model.",
        ".vision_model.",
        "visual.",
        ".visual.",
        "vision_tower.",
        ".vision_tower.",
        ".mtp.",
        "mtp.",
    )
    return not any(fragment in name for fragment in excluded_fragments)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
