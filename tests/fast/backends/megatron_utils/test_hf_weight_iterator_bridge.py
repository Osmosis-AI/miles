"""Tests for Megatron Bridge HF weight iterator helpers."""

from argparse import Namespace

import torch

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import _postprocess_bridge_lora_param


def test_bridge_lora_b_rank_first_tensor_is_transposed_for_sglang():
    args = Namespace(lora_rank=8)
    param = torch.randn(8, 4096)

    result = _postprocess_bridge_lora_param(
        args,
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        param,
    )

    assert result.shape == (4096, 8)
    assert torch.equal(result, param.transpose(-1, -2))
    assert result.is_contiguous()


def test_bridge_lora_b_peft_layout_is_left_unchanged():
    args = Namespace(lora_rank=8)
    param = torch.randn(4096, 8)

    result = _postprocess_bridge_lora_param(
        args,
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        param,
    )

    assert result is param


def test_bridge_lora_a_is_left_unchanged():
    args = Namespace(lora_rank=8)
    param = torch.randn(8, 2048)

    result = _postprocess_bridge_lora_param(
        args,
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        param,
    )

    assert result is param


def test_bridge_qwen3_5_gated_qkv_lora_b_drops_q_gate_rows():
    args = Namespace(
        attention_output_gate=True,
        hidden_size=16,
        kv_channels=2,
        lora_rank=1,
        num_attention_heads=4,
        num_query_groups=2,
    )
    param = torch.arange(24, dtype=torch.float32).view(24, 1)

    result = _postprocess_bridge_lora_param(
        args,
        "model.layers.0.self_attn.qkv_proj.lora_B.weight",
        param,
    )

    qgkv = param.view(2, 6, 2, 1)
    q_with_gate, k, v = qgkv.split([4, 1, 1], dim=1)
    q = q_with_gate.view(2, 2, 2, 2, 1)[:, 0]
    expected = torch.cat(
        [
            q.reshape(-1, 1),
            k.reshape(-1, 1),
            v.reshape(-1, 1),
        ],
        dim=0,
    )

    assert result.shape == (16, 1)
    assert torch.equal(result, expected)
    assert result.is_contiguous()


def test_bridge_qwen3_5_gated_qkv_lora_b_handles_rank_first_export():
    args = Namespace(
        attention_output_gate=True,
        hidden_size=16,
        kv_channels=2,
        lora_rank=1,
        num_attention_heads=4,
        num_query_groups=2,
    )
    param = torch.arange(24, dtype=torch.float32).view(1, 24)

    result = _postprocess_bridge_lora_param(
        args,
        "model.layers.0.self_attn.qkv_proj.lora_B.weight",
        param,
    )

    assert result.shape == (16, 1)
    assert result.is_contiguous()


def test_bridge_qwen3_5_gated_q_lora_b_drops_q_gate_rows():
    args = Namespace(
        attention_output_gate=True,
        hidden_size=16,
        kv_channels=2,
        lora_rank=1,
        num_attention_heads=4,
        num_query_groups=2,
    )
    param = torch.arange(16, dtype=torch.float32).view(16, 1)

    result = _postprocess_bridge_lora_param(
        args,
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        param,
    )

    expected = param.view(2, 2, 2, 2, 1)[:, 0].reshape(-1, 1)

    assert result.shape == (8, 1)
    assert torch.equal(result, expected)
    assert result.is_contiguous()
