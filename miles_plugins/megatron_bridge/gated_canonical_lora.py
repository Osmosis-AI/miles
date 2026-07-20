"""Gate-aware CanonicalLoRA for attention-output-gate models (e.g. Qwen3.5).

Megatron-Bridge sizes the q adapter as ``kv_channels * num_attention_heads``,
correct only for ungated attention. Gated attention's ``q_proj`` outputs query +
output gate (2x per head), so the ungated q adapter exports a ``q_proj`` LoRA of
the wrong shape and SGLang's fused qkv LoRA buffer cannot load it. The transform
wrapper doubles the head count seen by the bridge while it builds a gated
``linear_qkv`` adapter; everything else delegates to the current bridge code.
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


def install() -> None:
    import megatron.bridge.peft.canonical_lora as cl

    bridge_transform = cl.CanonicalLoRA.transform

    def transform_gated(self, m, name=None, prefix=None):
        # No adapters on MTP layers: the rollout engine never runs MTP, and the
        # bridge exporter has no HF mappings for them (asserts at weight sync).
        if prefix and ".mtp." in f".{prefix}.":
            return m
        config = getattr(m, "config", None)
        if name == "linear_qkv" and getattr(config, "attention_output_gate", False):
            # q_out_features = kv_channels * num_attention_heads inside the bridge;
            # double the head count so the q adapter covers query + gate.
            original_heads = config.num_attention_heads
            config.num_attention_heads = 2 * original_heads
            try:
                return bridge_transform(self, m, name=name, prefix=prefix)
            finally:
                config.num_attention_heads = original_heads
        return bridge_transform(self, m, name=name, prefix=prefix)

    cl.CanonicalLoRA.transform = transform_gated
    cl.LoRALinearSplitQKV._interleave_qkv = interleave_qkv_gated
    logger.info("Installed gate-aware CanonicalLoRA transform and qkv interleave")
