"""Diagnostic: per-sample log prob alignment between Megatron recomputation and SGLang rollout.

This is a custom loss function wrapper that logs detailed per-sample/per-token
alignment data to help debug the growing train_rollout_logprob_abs_diff metric
in multi-LoRA training.

Usage:
    Add to your training config:
        --custom-megatron-before-train-step-hook-path \
            examples.multi_lora.diagnose_logprob_diff.before_train_step_hook

    The hook patches the loss function for one step, collecting:
      1. Per-sample: max & mean per-token abs diff, adapter slot, response length
      2. Per-adapter: weight L2 norms for LoRA A and B
      3. First & last 3 tokens of old_log_probs and rollout_log_probs (alignment check)

    Output is logged at INFO level with prefix [logprob_diag].
"""

import logging
import torch

logger = logging.getLogger(__name__)

_DIAG_INTERVAL = 5  # log every N steps; set to 1 for every step
_STEP_COUNT = 0


def before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Called before each training step. Patches loss function to add diagnostics."""
    global _STEP_COUNT
    _STEP_COUNT += 1
    if _STEP_COUNT % _DIAG_INTERVAL != 1:
        return

    from miles.backends.training_utils import loss as loss_mod
    _original_policy_loss = loss_mod.policy_loss_function

    def _diag_policy_loss(args, batch, logits, sum_of_sample_mean):
        loss, reported = _original_policy_loss(args, batch, logits, sum_of_sample_mean)

        if "rollout_log_probs" not in batch or not batch["rollout_log_probs"]:
            return loss, reported

        old_lp = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
        rollout_lp = batch["rollout_log_probs"]
        response_lengths = batch["response_lengths"]
        adapter_slots = batch.get("adapter_slots")

        try:
            _run_diagnostics(old_lp, rollout_lp, response_lengths, adapter_slots, model, args)
        except Exception as e:
            logger.warning(f"[logprob_diag] diagnostic failed: {e}")

        # Unpatch after one micro-batch
        loss_mod.policy_loss_function = _original_policy_loss
        return loss, reported

    loss_mod.policy_loss_function = _diag_policy_loss


def _run_diagnostics(old_lp_list, rollout_lp_list, response_lengths, adapter_slots, model, args):
    """Log per-sample alignment data."""
    n_samples = len(old_lp_list)
    per_adapter_stats = {}

    for i in range(n_samples):
        old_t = old_lp_list[i]
        roll_t = rollout_lp_list[i]
        rlen = response_lengths[i]
        slot = adapter_slots[i] if adapter_slots is not None else -1

        # Sanity: both tensors should have the same length = response_length
        if old_t.shape[0] != roll_t.shape[0]:
            logger.error(
                f"[logprob_diag] SHAPE MISMATCH sample {i}: "
                f"old={old_t.shape[0]} rollout={roll_t.shape[0]} rlen={rlen}"
            )
            continue

        diff = (old_t - roll_t).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Log first & last 3 token values for alignment check
        n_show = min(3, old_t.shape[0])
        first_old = old_t[:n_show].tolist()
        first_roll = roll_t[:n_show].tolist()
        last_old = old_t[-n_show:].tolist()
        last_roll = roll_t[-n_show:].tolist()

        if slot not in per_adapter_stats:
            per_adapter_stats[slot] = {
                "max_diffs": [], "mean_diffs": [], "rlens": [], "count": 0,
            }
        per_adapter_stats[slot]["max_diffs"].append(max_diff)
        per_adapter_stats[slot]["mean_diffs"].append(mean_diff)
        per_adapter_stats[slot]["rlens"].append(rlen)
        per_adapter_stats[slot]["count"] += 1

        # Log detailed info for first 2 samples per adapter
        if per_adapter_stats[slot]["count"] <= 2:
            logger.info(
                f"[logprob_diag] sample={i} slot={slot} rlen={rlen} "
                f"max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}\n"
                f"  first3_old={[f'{v:.6f}' for v in first_old]}\n"
                f"  first3_rol={[f'{v:.6f}' for v in first_roll]}\n"
                f"  last3_old ={[f'{v:.6f}' for v in last_old]}\n"
                f"  last3_rol ={[f'{v:.6f}' for v in last_roll]}"
            )

    # Per-adapter summary
    for slot, stats in sorted(per_adapter_stats.items()):
        max_of_max = max(stats["max_diffs"])
        avg_mean = sum(stats["mean_diffs"]) / len(stats["mean_diffs"])
        avg_rlen = sum(stats["rlens"]) / len(stats["rlens"])
        logger.info(
            f"[logprob_diag] ADAPTER slot={slot}: "
            f"n_samples={stats['count']} avg_rlen={avg_rlen:.1f} "
            f"avg_mean_diff={avg_mean:.6e} max_of_max_diff={max_of_max:.6e}"
        )

    # Log adapter weight norms (just first MultiLoRA layer for brevity)
    try:
        from megatron.bridge.peft.multi_lora_layers import _iter_multi_lora_modules
        models = model if isinstance(model, list) else [model]
        for m in _iter_multi_lora_modules(models):
            for idx, adapter in enumerate(m.adapters):
                a_norm = adapter.linear_in.weight.data.float().norm().item()
                b_norm = adapter.linear_out.weight.data.float().norm().item()
                logger.info(
                    f"[logprob_diag] WEIGHTS layer0 adapter={idx}: "
                    f"A_norm={a_norm:.6f} B_norm={b_norm:.6f} "
                    f"alpha={m.alpha_values[idx].item():.1f} rank={m.rank_values[idx].item():.0f}"
                )
            break  # just first layer
    except Exception as e:
        logger.warning(f"[logprob_diag] weight norm logging failed: {e}")
