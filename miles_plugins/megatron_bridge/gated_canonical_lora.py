"""Gate-aware q sizing for CanonicalLoRA.

Megatron-Bridge's ``CanonicalLoRA.transform`` sizes the q adapter as
``ParallelLinearAdapter(in_features, in_features)`` — correct only when
``num_attention_heads * head_dim == hidden_size``. Qwen3.5 uses GATED attention,
so ``q_proj`` outputs ``2 * num_attention_heads * head_dim`` (query + output gate)
= 8192 != hidden 2048. The undersized q adapter (2048) makes the exported
``q_proj`` LoRA the wrong shape, so SGLang's fused qkv LoRA is misapplied and
rollouts come out as gibberish.

This patches only the q out_features to be gate-aware. k/v
(``kv_channels * num_query_groups``) and fc1 up/gate (``out_features // 2``) are
already correct. The formula reduces to the original ``in_features`` for
non-gated square-attention models, so it is a safe generalization.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _patched_transform(self, m, name=None, prefix=None):
    import megatron.bridge.peft.canonical_lora as cl
    from torch import nn

    if isinstance(m, (cl.LinearAdapter, cl.LoRALinear, cl.LoRALinearSplitQKV, cl.LoRALinearSplitFC1UpGate)):
        return m

    ans = self.match(m, name, prefix)
    if ans is None:
        return m
    match, full_name = ans

    if isinstance(m, nn.Linear):
        return cl.LinearAdapter(
            m, dim=self.dim, alpha=self.alpha, dropout=self.dropout, lora_A_init_method=self.lora_A_init_method
        )

    is_expert = cl.is_expert_linear(full_name)
    input_is_parallel, in_features, out_features, disable_sp_comm, base_linear_is_parallel = (
        cl.get_adapter_attributes_from_linear(m, is_expert=is_expert)
    )
    adapter_kwargs = dict(
        dim=self.dim,
        base_linear_name=full_name,
        activation="identity",
        norm_type=None,
        column_init_method=self.lora_A_init_method,
        row_init_method=self.lora_B_init_method,
        gather_output=False,
        input_is_parallel=input_is_parallel,
        dropout=self.dropout,
        dropout_position=self.dropout_position,
        model_parallel_config=getattr(m, "config", None),
        alpha=self.alpha,
        is_expert=is_expert,
        disable_sequence_parallel_comm=disable_sp_comm,
        base_linear_is_parallel=base_linear_is_parallel,
    )

    if name in ["linear_proj", "linear_fc2"]:
        adapter = cl.ParallelLinearAdapter(in_features, out_features, **adapter_kwargs)
        return cl.LoRALinear(m, adapter)

    canonical_submodules = self.canonical_mapping[match]
    if name == "linear_qkv":
        adapter_q = adapter_k = adapter_v = None
        kv_out_features = m.config.kv_channels * m.config.num_query_groups
        # gate-aware q size: query+gate when attention_output_gate (Qwen3.5)
        q_out_features = m.config.num_attention_heads * m.config.kv_channels
        if getattr(m.config, "attention_output_gate", False):
            q_out_features *= 2
        if "linear_q" in canonical_submodules:
            adapter_q = cl.ParallelLinearAdapter(in_features, q_out_features, **adapter_kwargs)
        if "linear_k" in canonical_submodules:
            adapter_k = cl.ParallelLinearAdapter(in_features, kv_out_features, **adapter_kwargs)
        if "linear_v" in canonical_submodules:
            adapter_v = cl.ParallelLinearAdapter(in_features, kv_out_features, **adapter_kwargs)
        return cl.LoRALinearSplitQKV(m, nn.ModuleDict({"adapter_q": adapter_q, "adapter_k": adapter_k, "adapter_v": adapter_v}))

    if name == "linear_fc1":
        adapter_up = adapter_gate = None
        if "linear_fc1_up" in canonical_submodules:
            adapter_up = cl.ParallelLinearAdapter(in_features, out_features // 2, **adapter_kwargs)
        if "linear_fc1_gate" in canonical_submodules:
            adapter_gate = cl.ParallelLinearAdapter(in_features, out_features // 2, **adapter_kwargs)
        return cl.LoRALinearSplitFC1UpGate(m, nn.ModuleDict({"adapter_up": adapter_up, "adapter_gate": adapter_gate}))

    return m


def install() -> None:
    import megatron.bridge.peft.canonical_lora as cl

    cl.CanonicalLoRA.transform = _patched_transform
    logger.info("Installed gate-aware q sizing for CanonicalLoRA.transform")
