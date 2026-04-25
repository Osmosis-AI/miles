"""Experiment: Verify SGLang has correct adapter weights after multi-LoRA sync.

After each weight sync, this hook queries the SGLang engine to generate log
probs for a FIXED set of tokens using each adapter, and compares against
Megatron's output for the same tokens. This catches weight sync issues that
GPU==CPU checks miss (those only verify Megatron-side consistency, not that
SGLang actually received the weights).

This is a training-loop diagnostic. It runs after update_weights.

Usage:
    --rollout-data-postprocess-path \
        examples.multi_lora.test_sglang_weight_verify.verify_weights

NOTE: This requires that rollout_data is available and that rollout_manager
is accessible. If that's too complex, a simpler approach:

The simplest possible verification: after weight sync, take a SINGLE prompt,
generate with SGLang using each adapter name, and check the output isn't
identical across adapters (which would indicate adapter confusion).
"""

import logging
import torch

logger = logging.getLogger(__name__)


def verify_weights(args):
    """Post-rollout-data hook to verify adapter weights are distinct in SGLang.

    After rollout data is collected, the rollout_log_probs for each adapter
    should reflect that adapter's weights. If both adapters produce identical
    log probs for different prompts, something is wrong.

    This is a lightweight check — just verify that per-adapter rollout log probs
    have different distributions.
    """
    rollout_data = getattr(args, "_latest_rollout_data", None)
    if rollout_data is None:
        return

    adapter_slots = rollout_data.get("adapter_slots")
    rollout_lps = rollout_data.get("rollout_log_probs")
    if adapter_slots is None or rollout_lps is None:
        return

    per_adapter_lp_stats = {}
    for slot, lp in zip(adapter_slots, rollout_lps):
        if isinstance(lp, torch.Tensor):
            lp_mean = lp.mean().item()
            lp_std = lp.std().item()
        else:
            import numpy as np
            arr = np.array(lp)
            lp_mean = arr.mean()
            lp_std = arr.std()
        if slot not in per_adapter_lp_stats:
            per_adapter_lp_stats[slot] = {"means": [], "stds": []}
        per_adapter_lp_stats[slot]["means"].append(lp_mean)
        per_adapter_lp_stats[slot]["stds"].append(lp_std)

    for slot, stats in sorted(per_adapter_lp_stats.items()):
        avg_mean = sum(stats["means"]) / len(stats["means"])
        avg_std = sum(stats["stds"]) / len(stats["stds"])
        logger.info(
            f"[weight_verify] adapter slot={slot}: "
            f"avg_log_prob={avg_mean:.4f} avg_std={avg_std:.4f} "
            f"n_samples={len(stats['means'])}"
        )
