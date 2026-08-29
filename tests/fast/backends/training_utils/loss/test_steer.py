"""Tests for STEER token-level entropy-change reweighting.

Reference: "Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective"
(ACL 2026). The weight is lambda = exp(-alpha * |omega| / max|omega|) with
alpha = -log(lambda_min), so weights decay monotonically from 1 to lambda_min as the
estimated token-level entropy change grows.
"""

from __future__ import annotations

import math

import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_steer_weight

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state

LAMBDA_MIN = 0.7


def _weights(
    *,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    ppo_kl: torch.Tensor | None = None,
    clipfrac: torch.Tensor | None = None,
    active_tokens: torch.Tensor | None = None,
    lambda_min: float = LAMBDA_MIN,
) -> torch.Tensor:
    make_parallel_state()
    return compute_steer_weight(
        log_probs=log_probs,
        ppo_kl=torch.zeros_like(log_probs) if ppo_kl is None else ppo_kl,
        advantages=advantages,
        entropy=entropy,
        clipfrac=torch.zeros_like(log_probs) if clipfrac is None else clipfrac,
        active_tokens=torch.ones_like(log_probs, dtype=torch.bool) if active_tokens is None else active_tokens,
        lambda_min=lambda_min,
    )


@pytest.fixture
def sample() -> dict[str, torch.Tensor]:
    """A spread of token probabilities and both advantage signs."""
    probs = torch.tensor([0.05, 0.2, 0.5, 0.8, 0.95])
    return {
        "log_probs": probs.log(),
        "advantages": torch.tensor([1.0, -1.0, 2.0, -0.5, 1.0]),
        "entropy": torch.tensor([2.0, 1.5, 1.0, 0.5, 0.2]),
    }


def test_weights_lie_in_expected_range(sample):
    weights = _weights(**sample)
    assert torch.all(weights >= LAMBDA_MIN - 1e-6)
    assert torch.all(weights <= 1.0 + 1e-6)


def test_largest_entropy_change_attains_lambda_min(sample):
    """The batch-max |omega| token is attenuated exactly to lambda_min."""
    weights = _weights(**sample)
    assert weights.min().item() == pytest.approx(LAMBDA_MIN, abs=1e-6)


def test_weight_decreases_monotonically_with_entropy_change(sample):
    """Ranking by weight must be the exact reverse of ranking by |omega|."""
    probs = sample["log_probs"].exp()
    omega = sample["advantages"] * probs * (1 - probs) * (sample["log_probs"] + sample["entropy"])
    weights = _weights(**sample)

    order_by_omega = torch.argsort(omega.abs(), descending=True)
    order_by_weight = torch.argsort(weights, descending=False)
    assert torch.equal(order_by_omega, order_by_weight)


def test_matches_closed_form(sample):
    """Weights equal exp(-alpha * |omega| / max|omega|) for the paper's omega."""
    probs = sample["log_probs"].exp()
    omega = sample["advantages"] * probs * (1 - probs) * (sample["log_probs"] + sample["entropy"])
    alpha = -math.log(LAMBDA_MIN)
    expected = torch.exp(-alpha * omega.abs() / omega.abs().amax())

    torch.testing.assert_close(_weights(**sample), expected, rtol=1e-5, atol=1e-6)


def test_zero_advantage_group_is_a_noop(sample):
    """A group where every sample earned the same reward must not be reweighted.

    GRPO yields all-zero advantages there, so omega is identically zero and the
    normalizer would be 0/0 without the clamp.
    """
    weights = _weights(**{**sample, "advantages": torch.zeros_like(sample["advantages"])})
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_clipped_tokens_are_not_attenuated(sample):
    """Clipping zeroes a token's gradient, so it cannot change entropy (I_clip = 0)."""
    clipfrac = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    weights = _weights(**sample, clipfrac=clipfrac)
    assert weights[2].item() == pytest.approx(1.0, abs=1e-6)


def test_inactive_tokens_are_excluded_from_the_normalizer(sample):
    """A masked-out token must not be able to inflate max|omega|."""
    active = torch.tensor([True, True, False, True, True])
    weights = _weights(**sample, active_tokens=active)

    # With the dominant token masked out, some *active* token now attains lambda_min.
    assert weights[2].item() == pytest.approx(1.0, abs=1e-6)
    assert weights[active].min().item() == pytest.approx(LAMBDA_MIN, abs=1e-6)


def test_lambda_min_one_disables_reweighting(sample):
    weights = _weights(**sample, lambda_min=1.0)
    torch.testing.assert_close(weights, torch.ones_like(weights))


def _run_policy_loss(**arg_overrides):
    """Run policy_loss_function end-to-end on deterministic inputs."""
    parallel_state = make_parallel_state()
    args = make_args(loss_type="policy_loss", advantage_estimator="grpo", **arg_overrides)
    inputs = make_inputs(
        seed=0,
        batch_size=3,
        prompt_lens=[4, 6, 5],
        response_lens=[7, 5, 6],
        vocab_size=32,
        args=args,
    )
    batch = make_batch(inputs, "policy_loss")
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )
    loss, metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    return loss, metrics, parallel_state


def test_policy_loss_reports_steer_weight_only_when_enabled():
    num_samples = 3
    _, off_metrics, _ = _run_policy_loss(use_steer=False)
    _, on_metrics, _ = _run_policy_loss(use_steer=True, steer_lambda_min=LAMBDA_MIN)

    assert "steer_weight" not in off_metrics
    assert "steer_weight" in on_metrics
    # Reported metrics are sums of per-sample means; `aggregate_train_losses` divides by
    # the sample count, so the logged value is the mean weight and lies in [lambda_min, 1].
    mean_weight = on_metrics["steer_weight"].item() / num_samples
    assert LAMBDA_MIN - 1e-6 <= mean_weight <= 1.0 + 1e-6


def test_policy_loss_changes_when_steer_enabled():
    off_loss, _, _ = _run_policy_loss(use_steer=False)
    on_loss, _, _ = _run_policy_loss(use_steer=True, steer_lambda_min=LAMBDA_MIN)
    assert not torch.allclose(off_loss, on_loss)


def test_steer_forces_entropy_computation():
    """STEER needs per-token entropy even with entropy_coef=0 and observation off."""
    _, metrics, _ = _run_policy_loss(
        use_steer=True,
        steer_lambda_min=LAMBDA_MIN,
        entropy_coef=0.0,
        observe_training_entropy=False,
    )
    assert metrics["entropy_loss"].item() != 0.0


def test_lambda_min_one_is_a_noop_end_to_end():
    """lambda_min=1 makes every weight 1, so the loss must match the disabled path."""
    off_loss, _, _ = _run_policy_loss(use_steer=False)
    on_loss, _, _ = _run_policy_loss(use_steer=True, steer_lambda_min=1.0)
    torch.testing.assert_close(off_loss, on_loss)


def test_importance_ratio_scales_entropy_change(sample):
    """omega is linear in the importance ratio, so ranking follows r * A * delta."""
    ppo_kl = torch.tensor([0.0, 0.0, -math.log(2.0), 0.0, 0.0])  # ratio = 2 on token 2
    probs = sample["log_probs"].exp()
    omega = sample["advantages"] * probs * (1 - probs) * (sample["log_probs"] + sample["entropy"])
    omega[2] *= 2.0
    alpha = -math.log(LAMBDA_MIN)
    expected = torch.exp(-alpha * omega.abs() / omega.abs().amax())

    torch.testing.assert_close(_weights(**sample, ppo_kl=ppo_kl), expected, rtol=1e-5, atol=1e-6)
