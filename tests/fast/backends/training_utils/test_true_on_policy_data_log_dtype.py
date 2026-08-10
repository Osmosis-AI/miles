from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils import log_utils
from miles.backends.training_utils import mm_data


def _get_cp2_rollout_data(monkeypatch, response_fields):
    parallel_state = SimpleNamespace(
        intra_dp=SimpleNamespace(rank=0, size=1),
        effective_dp=SimpleNamespace(rank=0, size=1),
        cp=SimpleNamespace(rank=0, size=2),
    )
    rollout_data = {
        "tokens": [[10, 11, 12, 13, 14, 15, 16, 17]],
        "loss_masks": [[1, 1, 1, 1, 1, 1]],
        "total_lengths": [8],
        "response_lengths": [6],
        **response_fields,
    }

    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *_args, **_kwargs: (rollout_data, None))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    args = Namespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        bf16=True,
        fp16=False,
        enable_witness=False,
    )
    result, _store_get_result = data_utils.get_rollout_data(args, None)
    return result


def test_true_on_policy_rollout_logprob_dtype_follows_training_precision():
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=True, fp16=False)) is torch.bfloat16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=False, fp16=True)) is torch.float16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=False, bf16=True, fp16=False)) is torch.float32
    )


def test_true_on_policy_log_checker_passes_when_values_and_dtype_match(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )

    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda metric_name, args, rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
        "rollout_log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
    }

    log_utils.log_rollout_data(
        1,
        Namespace(
            ci_test=True,
            ci_disable_logprobs_checker=False,
            true_on_policy_mode=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert captured["log_dict"]["log_probs"] == captured["log_dict"]["rollout_log_probs"]


def test_get_rollout_data_cp2_slices_sampled_opd_fields(monkeypatch):
    rollout_data = _get_cp2_rollout_data(
        monkeypatch,
        {
            "rollout_log_probs": [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
            "teacher_log_probs": [
                torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=torch.float64)
            ],
            "opd_loss_masks": [torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.int64)],
        },
    )

    torch.testing.assert_close(rollout_data["rollout_log_probs"][0], torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(rollout_data["teacher_log_probs"][0], torch.tensor([10.0, 15.0]))
    torch.testing.assert_close(rollout_data["opd_loss_masks"][0], torch.tensor([0.0, 0.0]))
    for key in ("teacher_log_probs", "opd_loss_masks"):
        assert rollout_data[key][0].dtype is torch.float32
        assert rollout_data[key][0].device == rollout_data["tokens"][0].device


def test_get_rollout_data_cp2_slices_precomputed_opd_field(monkeypatch):
    rollout_data = _get_cp2_rollout_data(
        monkeypatch,
        {
            "rollout_log_probs": [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
            "opd_reverse_kl": [
                torch.tensor([20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype=torch.float64)
            ],
        },
    )

    torch.testing.assert_close(rollout_data["rollout_log_probs"][0], torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(rollout_data["opd_reverse_kl"][0], torch.tensor([20.0, 25.0]))
    assert rollout_data["opd_reverse_kl"][0].dtype is torch.float32
    assert rollout_data["opd_reverse_kl"][0].device == rollout_data["tokens"][0].device


def test_multimodal_expansion_regathers_and_reslices_cp2_opd_fields(monkeypatch):
    rollout_data = _get_cp2_rollout_data(
        monkeypatch,
        {
            "tokens": [[mm_data.KIMI_VL_MEDIA_TOKEN_ID, 11, 12, 13, 14, 15, 16, 17]],
            "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
            "rollout_log_probs": [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
            "teacher_log_probs": [
                torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=torch.float64)
            ],
            "opd_reverse_kl": [
                torch.tensor([20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype=torch.float64)
            ],
            "opd_loss_masks": [torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.int64)],
        },
    )
    expected_local_values = iter(
        (
            torch.tensor([0.0, 5.0]),
            torch.tensor([10.0, 15.0]),
            torch.tensor([20.0, 25.0]),
            torch.tensor([0.0, 0.0]),
        )
    )
    full_values = iter(
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            torch.tensor([20.0, 21.0, 22.0, 23.0, 24.0, 25.0]),
            torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        )
    )
    gather_lengths = []

    def fake_all_gather(value, total_length, response_length):
        torch.testing.assert_close(value, next(expected_local_values))
        gather_lengths.append((total_length, response_length))
        return next(full_values)

    monkeypatch.setattr(mm_data, "get_parallel_state", data_utils.get_parallel_state)
    monkeypatch.setattr(mm_data, "all_gather_with_cp", fake_all_gather)

    mm_data.expand_multimodal_rollout_data_in_place(rollout_data, qkv_format="thd")

    assert gather_lengths == [(8, 6)] * 4
    assert rollout_data["total_lengths"] == [11]
    assert rollout_data["response_lengths"] == [6]
    torch.testing.assert_close(rollout_data["rollout_log_probs"][0], torch.tensor([5.0]))
    torch.testing.assert_close(rollout_data["teacher_log_probs"][0], torch.tensor([15.0]))
    torch.testing.assert_close(rollout_data["opd_reverse_kl"][0], torch.tensor([25.0]))
    torch.testing.assert_close(rollout_data["opd_loss_masks"][0], torch.tensor([0.0]))


def test_multimodal_response_expansion_zero_fills_opd_fields_cp1(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=1))
    monkeypatch.setattr(mm_data, "get_parallel_state", lambda: parallel_state)

    rollout_data = {
        "tokens": [torch.tensor([10, 11, mm_data.KIMI_VL_MEDIA_TOKEN_ID, 13, 14])],
        "loss_masks": [torch.tensor([1, 1, 1])],
        "total_lengths": [5],
        "response_lengths": [3],
        "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
        "rollout_log_probs": [torch.tensor([1.0, 2.0, 3.0])],
        "teacher_log_probs": [torch.tensor([11.0, 12.0, 13.0])],
        "opd_reverse_kl": [torch.tensor([21.0, 22.0, 23.0])],
        "opd_loss_masks": [torch.tensor([1.0, 1.0, 0.0])],
    }

    mm_data.expand_multimodal_rollout_data_in_place(rollout_data, qkv_format="thd")

    assert rollout_data["total_lengths"] == [8]
    assert rollout_data["response_lengths"] == [6]
    torch.testing.assert_close(rollout_data["loss_masks"][0], torch.tensor([0, 0, 0, 0, 1, 1]))
    torch.testing.assert_close(rollout_data["rollout_log_probs"][0], torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0, 3.0]))
    torch.testing.assert_close(
        rollout_data["teacher_log_probs"][0],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 12.0, 13.0]),
    )
    torch.testing.assert_close(
        rollout_data["opd_reverse_kl"][0],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 22.0, 23.0]),
    )
    torch.testing.assert_close(
        rollout_data["opd_loss_masks"][0],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )


def test_multimodal_response_expansion_regathers_zero_fills_and_reslices_cp2(monkeypatch):
    rollout_data = _get_cp2_rollout_data(
        monkeypatch,
        {
            "tokens": [[10, 11, mm_data.KIMI_VL_MEDIA_TOKEN_ID, 13, 14, 15, 16, 17]],
            "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
            "rollout_log_probs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            "teacher_log_probs": [torch.tensor([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])],
            "opd_reverse_kl": [torch.tensor([21.0, 22.0, 23.0, 24.0, 25.0, 26.0])],
            "opd_loss_masks": [torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])],
        },
    )
    expected_local_values = iter(
        (
            torch.tensor([1.0, 6.0]),
            torch.tensor([11.0, 16.0]),
            torch.tensor([21.0, 26.0]),
            torch.tensor([1.0, 0.0]),
        )
    )
    full_values = iter(
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            torch.tensor([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            torch.tensor([21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        )
    )

    def fake_all_gather(value, total_length, response_length):
        torch.testing.assert_close(value, next(expected_local_values))
        assert (total_length, response_length) == (8, 6)
        return next(full_values)

    monkeypatch.setattr(mm_data, "get_parallel_state", data_utils.get_parallel_state)
    monkeypatch.setattr(mm_data, "all_gather_with_cp", fake_all_gather)

    mm_data.expand_multimodal_rollout_data_in_place(rollout_data, qkv_format="thd")

    assert rollout_data["total_lengths"] == [11]
    assert rollout_data["response_lengths"] == [9]
    torch.testing.assert_close(rollout_data["rollout_log_probs"][0], torch.tensor([0.0, 0.0, 6.0]))
    torch.testing.assert_close(rollout_data["teacher_log_probs"][0], torch.tensor([0.0, 0.0, 16.0]))
    torch.testing.assert_close(rollout_data["opd_reverse_kl"][0], torch.tensor([0.0, 0.0, 26.0]))
    torch.testing.assert_close(rollout_data["opd_loss_masks"][0], torch.tensor([0.0, 0.0, 0.0]))


def test_multimodal_expansion_rejects_bshd_with_context_parallelism(monkeypatch):
    parallel_state = SimpleNamespace(cp=SimpleNamespace(size=2))
    monkeypatch.setattr(mm_data, "get_parallel_state", lambda: parallel_state)
    rollout_data = {
        "tokens": [torch.tensor([mm_data.KIMI_VL_MEDIA_TOKEN_ID, 11, 12, 13, 14, 15, 16, 17])],
        "loss_masks": [torch.tensor([1, 1, 1, 1, 1, 1])],
        "total_lengths": [8],
        "response_lengths": [6],
        "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
    }

    with pytest.raises(ValueError, match="requires qkv_format='thd'"):
        mm_data.expand_multimodal_rollout_data_in_place(rollout_data, qkv_format="bshd")
