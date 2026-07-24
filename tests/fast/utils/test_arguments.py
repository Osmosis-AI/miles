import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.sglang_utils.arguments import validate_args as validate_sglang_args
from miles.utils.arguments import (
    _maybe_apply_dumper_overrides,
    _set_colocate_sglang_cuda_graph_defaults,
    _validate_opd_args,
    get_miles_extra_args_provider,
)
from miles.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]
CROSS_VOCAB_RM_PATH = "miles.rollout.on_policy_distillation.reward_func_cross_vocab"
CROSS_VOCAB_POST_PROCESS_PATH = (
    "miles.rollout.on_policy_distillation.post_process_rewards_cross_vocab"
)


def make_opd_args(**overrides):
    values = {
        "custom_rm_path": None,
        "custom_reward_post_process_path": None,
        "use_opd": False,
        "opd_type": None,
        "opd_log_prob_top_k": 0,
        "opd_teacher_load": None,
        "teacher_tokenizer_path": None,
        "opd_teacher_timeout": 300.0,
        "opd_teacher_retries": 2,
        "opd_teacher_concurrency": 0,
        "apply_chat_template": False,
        "opd_prompt_messages_key": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save: str | None = "/tmp/checkpoint",
        save_interval: int | None = 5,
        save_retain_interval: int | None = 10,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save=save,
            save_interval=save_interval,
            save_retain_interval=save_retain_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.rollout_health_check_interval == 30.0
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save == "/tmp/checkpoint"
        assert args.save_interval == 5
        assert args.save_retain_interval == 10

    def test_disables_all_heartbeats(self) -> None:
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True
        assert args.rollout_health_check_interval == 1e18

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.start_rollout_id == 0
        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save is None
        assert args.save_interval is None
        assert args.save_retain_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6


def test_recompute_logprobs_via_prefill_flag_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--recompute-logprobs-via-prefill"] + REQUIRED_ARGS)

    assert args.recompute_logprobs_via_prefill is True


def test_colocate_parser_normalizes_sglang_server_args_across_versions():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--colocate"] + REQUIRED_ARGS)
    validate_sglang_args(args)
    _set_colocate_sglang_cuda_graph_defaults(args)

    assert args.colocate is True
    assert args.sglang_dp_size == 1
    assert args.sglang_pp_size == 1
    assert args.sglang_ep_size == 1
    if hasattr(args, "sglang_cuda_graph_backend_prefill"):
        assert args.sglang_cuda_graph_backend_prefill == "disabled"
    else:
        assert args.sglang_disable_piecewise_cuda_graph is True


def test_colocate_defaults_cuda_graph_backend_prefill_when_available():
    args = SimpleNamespace(sglang_cuda_graph_backend_prefill=None)

    _set_colocate_sglang_cuda_graph_defaults(args)

    assert args.sglang_cuda_graph_backend_prefill == "disabled"


def test_colocate_defaults_legacy_piecewise_cuda_graph_off():
    args = SimpleNamespace(
        sglang_disable_piecewise_cuda_graph=False,
        sglang_enforce_piecewise_cuda_graph=False,
    )

    _set_colocate_sglang_cuda_graph_defaults(args)

    assert args.sglang_disable_piecewise_cuda_graph is True


def test_colocate_warns_for_explicit_legacy_piecewise_cuda_graph(caplog):
    args = SimpleNamespace(
        sglang_disable_piecewise_cuda_graph=False,
        sglang_enforce_piecewise_cuda_graph=True,
    )

    _set_colocate_sglang_cuda_graph_defaults(args)

    assert args.sglang_disable_piecewise_cuda_graph is False
    assert "may trigger NVLS OOM" in caplog.text


def test_cross_vocab_opd_flags_are_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(
        [
            "--teacher-tokenizer-path",
            "/models/teacher",
            "--opd-prompt-messages-key",
            "opd_messages",
            "--opd-mask-teacher-logprob-tokens",
            "<think>",
            "</think>",
            "--opd-teacher-timeout",
            "12.5",
            "--opd-teacher-retries",
            "3",
            "--opd-teacher-concurrency",
            "4",
        ]
        + REQUIRED_ARGS
    )

    assert args.teacher_tokenizer_path == "/models/teacher"
    assert args.opd_prompt_messages_key == "opd_messages"
    assert args.opd_mask_teacher_logprob_tokens == ["<think>", "</think>"]
    assert args.opd_teacher_timeout == 12.5
    assert args.opd_teacher_retries == 3
    assert args.opd_teacher_concurrency == 4


@pytest.mark.parametrize(
    ("custom_rm_path", "custom_reward_post_process_path"),
    [
        (CROSS_VOCAB_RM_PATH, None),
        (None, CROSS_VOCAB_POST_PROCESS_PATH),
    ],
)
def test_cross_vocab_opd_requires_both_hooks(custom_rm_path, custom_reward_post_process_path):
    args = make_opd_args(
        custom_rm_path=custom_rm_path,
        custom_reward_post_process_path=custom_reward_post_process_path,
    )

    with pytest.raises(ValueError, match="requires both"):
        _validate_opd_args(args)


def test_cross_vocab_hooks_require_opd():
    args = make_opd_args(
        custom_rm_path=CROSS_VOCAB_RM_PATH,
        custom_reward_post_process_path=CROSS_VOCAB_POST_PROCESS_PATH,
    )

    with pytest.raises(ValueError, match="require --use-opd"):
        _validate_opd_args(args)


def test_cross_vocab_opd_requires_sglang():
    args = make_opd_args(
        custom_rm_path=CROSS_VOCAB_RM_PATH,
        custom_reward_post_process_path=CROSS_VOCAB_POST_PROCESS_PATH,
        use_opd=True,
        opd_type="megatron",
        teacher_tokenizer_path="/models/teacher",
    )

    with pytest.raises(ValueError, match="requires --opd-type=sglang"):
        _validate_opd_args(args)


def test_cross_vocab_opd_requires_teacher_tokenizer():
    args = make_opd_args(
        custom_rm_path=CROSS_VOCAB_RM_PATH,
        custom_reward_post_process_path=CROSS_VOCAB_POST_PROCESS_PATH,
        use_opd=True,
        opd_type="sglang",
    )

    with pytest.raises(ValueError, match="teacher-tokenizer-path"):
        _validate_opd_args(args)


def test_cross_vocab_opd_requires_raw_messages_with_chat_template():
    args = make_opd_args(
        custom_rm_path=CROSS_VOCAB_RM_PATH,
        custom_reward_post_process_path=CROSS_VOCAB_POST_PROCESS_PATH,
        use_opd=True,
        opd_type="sglang",
        teacher_tokenizer_path="/models/teacher",
        apply_chat_template=True,
    )

    with pytest.raises(ValueError, match="opd-prompt-messages-key"):
        _validate_opd_args(args)


def test_cross_vocab_opd_rejects_top_k_rewards():
    args = make_opd_args(
        custom_rm_path=CROSS_VOCAB_RM_PATH,
        custom_reward_post_process_path=CROSS_VOCAB_POST_PROCESS_PATH,
        use_opd=True,
        opd_type="sglang",
        opd_log_prob_top_k=8,
        teacher_tokenizer_path="/models/teacher",
    )

    with pytest.raises(ValueError, match="opd-log-prob-top-k=0"):
        _validate_opd_args(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("opd_teacher_timeout", 0, "timeout must be positive"),
        ("opd_teacher_retries", -1, "retries must be non-negative"),
        ("opd_teacher_concurrency", -1, "concurrency must be non-negative"),
    ],
)
def test_opd_teacher_request_limits_are_validated(field, value, message):
    args = make_opd_args(use_opd=True, opd_type="sglang", **{field: value})

    with pytest.raises(ValueError, match=message):
        _validate_opd_args(args)


def test_cross_vocab_opd_accepts_valid_sampled_token_configuration():
    args = make_opd_args(
        custom_rm_path=CROSS_VOCAB_RM_PATH,
        custom_reward_post_process_path=CROSS_VOCAB_POST_PROCESS_PATH,
        use_opd=True,
        opd_type="sglang",
        teacher_tokenizer_path="/models/teacher",
    )

    _validate_opd_args(args)
