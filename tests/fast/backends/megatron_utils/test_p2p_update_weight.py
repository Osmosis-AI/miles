from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p as p2p_mod
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin import (
    DistBucketedWeightUpdateMixin,
)


def _make_updater():
    updater = object.__new__(p2p_mod.UpdateWeightP2P)
    updater.rollout_engines = None
    updater._connection_stale = False
    updater._model_registered = False
    updater.transfer_plan = SimpleNamespace(
        _gathered_dp_rank=0,
        _rollout_num_gpus=1,
        plan_p2p=lambda: [],
    )
    return updater


def test_p2p_rejects_lora_at_construction() -> None:
    with pytest.raises(AssertionError, match="LoRA weight sync is not supported"):
        p2p_mod.UpdateWeightP2P(
            SimpleNamespace(),
            [],
            lambda: {},
            model_name="test-model",
            quantization_config=None,
            is_lora=True,
        )


def test_p2p_reconnect_refreshes_registration_and_freshness() -> None:
    updater = _make_updater()
    engine1, engine2 = MagicMock(name="transfer1"), MagicMock(name="transfer2")
    registry1, registry2 = {"first": object()}, {"second": object()}

    assert not updater.is_rollout_engines_fresh()
    with (
        patch.object(p2p_mod, "query_remote_weight_infos", return_value=({}, {}, {})),
        patch.object(p2p_mod, "create_transfer_engine", side_effect=[engine1, engine2]),
        patch.object(p2p_mod, "register_cpu_memory", side_effect=[registry1, registry2]) as register,
        patch.object(DistBucketedWeightUpdateMixin, "_pause_and_prepare_engines", return_value=None),
    ):
        updater.connect_rollout_engines([MagicMock()], MagicMock())
        shared1 = updater._shared_params_dict
        assert updater.is_rollout_engines_fresh()
        updater._pause_and_prepare_engines()
        assert updater._model_registered
        assert updater._weight_memory_registry is registry1

        updater.mark_engine_connection_stale()
        assert not updater.is_rollout_engines_fresh()
        updater.connect_rollout_engines([MagicMock()], MagicMock())
        shared2 = updater._shared_params_dict
        assert shared2 is not shared1
        assert updater.is_rollout_engines_fresh()
        assert not updater._model_registered
        updater._pause_and_prepare_engines()

    assert register.call_count == 2
    assert register.call_args_list[0].args == (shared1, engine1)
    assert register.call_args_list[1].args == (shared2, engine2)
    assert updater._weight_memory_registry is registry2


def test_p2p_failed_reconnect_remains_stale() -> None:
    updater = _make_updater()

    with (
        patch.object(p2p_mod, "query_remote_weight_infos", return_value=({}, {}, {})),
        patch.object(p2p_mod, "create_transfer_engine", return_value=MagicMock()),
    ):
        updater.connect_rollout_engines([MagicMock()], MagicMock())

    updater.mark_engine_connection_stale()
    with patch.object(p2p_mod, "query_remote_weight_infos", side_effect=RuntimeError("reconnect failed")):
        with pytest.raises(RuntimeError, match="reconnect failed"):
            updater.connect_rollout_engines([MagicMock()], MagicMock())

    assert not updater.is_rollout_engines_fresh()
