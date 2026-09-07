import argparse

import pytest

from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args
from miles.utils.misc import function_registry


def parse_topk_args(custom_reward=None):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    argv = [
        "--rollout-batch-size", "32", "--num-rollout", "36",
        "--use-opd", "--opd-type", "sglang", "--rm-url", "http://teacher/generate",
        "--opd-log-prob-top-k", "32", "--opd-top-k-strategy", "intersection",
        "--fully-async", "--pause-generation-mode", "retract",
    ]
    if custom_reward:
        argv.extend(["--custom-rm-path", custom_reward])
    return parser.parse_args(argv)


def test_declared_reward_support_preserves_fully_async_topk(monkeypatch):
    monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)

    def reward():
        pass

    reward.provides_student_top_logprobs = True
    with function_registry.temporary("test:topk_reward", reward):
        args = parse_topk_args("test:topk_reward")
        miles_validate_args(args)
    assert args.opd_log_prob_top_k == 32
    assert args.opd_top_k_strategy == "intersection"
    assert args.rollout_function_path == "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"


@pytest.mark.parametrize("capability", [None, False, 1, "True"])
def test_undeclared_reward_cannot_enable_student_topk(monkeypatch, capability):
    monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)

    def reward():
        pass

    if capability is not None:
        reward.provides_student_top_logprobs = capability
    with function_registry.temporary("test:topk_reward", reward):
        with pytest.raises(ValueError, match="provides_student_top_logprobs"):
            miles_validate_args(parse_topk_args("test:topk_reward"))


def test_no_custom_reward_still_requires_supported_producer(monkeypatch):
    monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
    with pytest.raises(ValueError, match="provides_student_top_logprobs"):
        miles_validate_args(parse_topk_args())
