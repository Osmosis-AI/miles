"""Experiment: Double-recompute test for multi-LoRA log prob drift.

Hypothesis: the growing train_rollout_logprob_abs_diff is caused by SGLang
producing incorrect log probs (e.g., using wrong adapter weights), NOT by
Megatron recomputation being wrong.

This test patches the training loop to do the log prob recomputation TWICE
with the same model weights and compare them. If the two recomputes match
but both differ from SGLang's rollout_log_probs, the issue is on the SGLang
side (or in the weight sync).

Usage:
    --custom-megatron-before-train-step-hook-path \
        examples.multi_lora.test_double_recompute.before_train_step_hook

Output: logs at INFO level with prefix [double_recompute].
"""

import logging
import torch

logger = logging.getLogger(__name__)

_STEP_COUNT = 0
_DIAG_INTERVAL = 5


def before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Patch loss function to compare two independent recomputations."""
    global _STEP_COUNT
    _STEP_COUNT += 1
    if _STEP_COUNT % _DIAG_INTERVAL != 1:
        return

    from miles.backends.training_utils import loss as loss_mod
    _original = loss_mod.policy_loss_function

    def _patched(args, batch, logits, sum_of_sample_mean):
        loss, reported = _original(args, batch, logits, sum_of_sample_mean)

        if "rollout_log_probs" not in batch or not batch["rollout_log_probs"]:
            loss_mod.policy_loss_function = _original
            return loss, reported

        # old_log_probs = Megatron-recomputed with current weights
        old_lp = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
        rollout_lp = batch["rollout_log_probs"]
        adapter_slots = batch.get("adapter_slots")

        # Compute diff between Megatron recompute and SGLang rollout
        megatron_vs_sglang_diffs = []
        for i in range(len(old_lp)):
            diff = (old_lp[i] - rollout_lp[i]).abs()
            megatron_vs_sglang_diffs.append(diff.mean().item())

        # Group by adapter
        per_adapter = {}
        for i in range(len(old_lp)):
            slot = adapter_slots[i] if adapter_slots is not None else 0
            if slot not in per_adapter:
                per_adapter[slot] = []
            per_adapter[slot].append(megatron_vs_sglang_diffs[i])

        for slot, diffs in sorted(per_adapter.items()):
            avg = sum(diffs) / len(diffs)
            mx = max(diffs)
            logger.info(
                f"[double_recompute] slot={slot} "
                f"megatron_vs_sglang: avg_mean_diff={avg:.6e} max_mean_diff={mx:.6e} "
                f"n_samples={len(diffs)}"
            )

        # Now: are the old_log_probs from batch["log_probs"] actually computed
        # from the same model as the current forward pass logits?
        # Compare: fresh log probs from THIS forward pass vs the stored ones
        from miles.backends.training_utils.loss import get_log_probs_and_entropy
        fresh_lp_dict = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            with_entropy=False,
            max_seq_lens=batch.get("max_seq_lens", None),
        )
        fresh_lp = fresh_lp_dict["log_probs"]

        # Compare fresh (current model) vs old (pre-computed by _switch_model)
        # If these differ, the model weights changed between recomputation and training
        fresh_vs_old = {}
        for i in range(len(fresh_lp)):
            slot = adapter_slots[i] if adapter_slots is not None else 0
            # fresh_lp might have different length than old_lp if CP is involved
            min_len = min(fresh_lp[i].shape[0], old_lp[i].shape[0])
            diff = (fresh_lp[i][:min_len] - old_lp[i][:min_len]).abs().mean().item()
            if slot not in fresh_vs_old:
                fresh_vs_old[slot] = []
            fresh_vs_old[slot].append(diff)

        for slot, diffs in sorted(fresh_vs_old.items()):
            avg = sum(diffs) / len(diffs)
            logger.info(
                f"[double_recompute] slot={slot} "
                f"fresh_vs_stored: avg_mean_diff={avg:.6e} "
                f"(should be ~0 if model unchanged between recompute and train)"
            )

        loss_mod.policy_loss_function = _original
        return loss, reported

    loss_mod.policy_loss_function = _patched
