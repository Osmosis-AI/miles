"""Gate-aware q sizing for CanonicalLoRA.

Megatron-Bridge's ``CanonicalLoRA.transform`` sizes the q adapter as
``kv_channels * num_attention_heads`` (num heads x head dim), correct only for
ungated attention. Qwen3.5 uses GATED attention, so ``q_proj`` outputs
``2 * num_attention_heads * head_dim`` (query + output gate) = 8192. The ungated
q adapter exports a ``q_proj`` LoRA of the wrong shape, so SGLang's fused qkv
LoRA buffer cannot load it. This mirrors the bridge transform but doubles
``q_out_features`` when ``attention_output_gate`` is set.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def interleave_qkv_gated(self, query, key, value):
    """Gate-aware replacement for LoRALinearSplitQKV._interleave_qkv.

    The base ``_interleave_qkv`` sizes q heads from ``num_attention_heads`` and
    cannot place the gated query (2x heads) into Megatron's per-group qkv layout.
    This mirrors the bridge qkv merge: per kv-group emit ``[q (gated), k, v]``.
    """
    config = self.to_wrap.config
    head_dim = config.kv_channels
    gate = 2 if getattr(config, "attention_output_gate", False) else 1
    # The adapter outputs are TP-partitioned (ColumnParallelLinear,
    # gather_output=False), so group/head counts must come from the local
    # tensor shapes, not the global config.
    num_kv = key.shape[-1] // head_dim
    q_per_group = query.shape[-1] // (num_kv * gate * head_dim)
    lead = query.shape[:-1]
    q = (
        query.reshape(*lead, num_kv, q_per_group, gate, head_dim)
        .transpose(-3, -2)
        .reshape(*lead, num_kv, gate * q_per_group * head_dim)
    )
    k = key.reshape(*lead, num_kv, head_dim)
    v = value.reshape(*lead, num_kv, head_dim)
    return torch.cat([q, k, v], dim=-1).reshape(*lead, -1)


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
    attrs = cl.get_adapter_attributes_from_linear(m, is_expert=is_expert)

    adapter_kwargs = dict(
        dim=self.dim,
        base_linear_name=full_name,
        activation="identity",
        norm_type=None,
        column_init_method=self.lora_A_init_method,
        row_init_method=self.lora_B_init_method,
        gather_output=False,
        input_is_parallel=attrs.input_is_parallel,
        dropout=self.dropout,
        dropout_position=self.dropout_position,
        model_parallel_config=getattr(m, "config", None),
        alpha=self.alpha,
        is_expert=is_expert,
        disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
        disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
        base_linear_is_parallel=attrs.base_linear_is_parallel,
    )

    if name in ["linear_proj", "linear_fc2"]:
        adapter = cl.ParallelLinearAdapter(attrs.in_features, attrs.out_features, **adapter_kwargs)
        return cl.LoRALinear(m, adapter)

    canonical_submodules = self.canonical_mapping[match]
    if name == "linear_qkv":
        adapter_q = adapter_k = adapter_v = None
        kv_out_features = m.config.kv_channels * m.config.num_query_groups
        q_out_features = m.config.kv_channels * m.config.num_attention_heads
        if getattr(m.config, "attention_output_gate", False):
            q_out_features *= 2
        if "linear_q" in canonical_submodules:
            adapter_q = cl.ParallelLinearAdapter(attrs.in_features, q_out_features, **adapter_kwargs)
        if "linear_k" in canonical_submodules:
            adapter_k = cl.ParallelLinearAdapter(attrs.in_features, kv_out_features, **adapter_kwargs)
        if "linear_v" in canonical_submodules:
            adapter_v = cl.ParallelLinearAdapter(attrs.in_features, kv_out_features, **adapter_kwargs)
        return cl.LoRALinearSplitQKV(
            m, cl.ModuleDict({"adapter_q": adapter_q, "adapter_k": adapter_k, "adapter_v": adapter_v})
        )

    if name == "linear_fc1":
        adapter_up = adapter_gate = None
        if "linear_fc1_up" in canonical_submodules:
            adapter_up = cl.ParallelLinearAdapter(attrs.in_features, attrs.out_features // 2, **adapter_kwargs)
        if "linear_fc1_gate" in canonical_submodules:
            adapter_gate = cl.ParallelLinearAdapter(attrs.in_features, attrs.out_features // 2, **adapter_kwargs)
        return cl.LoRALinearSplitFC1UpGate(m, cl.ModuleDict({"adapter_up": adapter_up, "adapter_gate": adapter_gate}))

    return m


def install() -> None:
    import megatron.bridge.peft.canonical_lora as cl

    cl.CanonicalLoRA.transform = _patched_transform
    cl.LoRALinearSplitQKV._interleave_qkv = interleave_qkv_gated
    logger.info("Installed gate-aware q sizing for CanonicalLoRA.transform")
