import logging
import time

import pytest
import requests


def test_load_lora_adapter_uses_sglang_serialized_tensors_field():
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    captured = {}
    engine = SGLangEngine.__new__(SGLangEngine)
    engine._make_request = lambda endpoint, payload: captured.update(
        endpoint=endpoint, payload=payload
    )

    ranked_tensors = ["rank-0", "rank-1"]
    engine.load_lora_adapter_from_tensors(
        lora_name="adapter",
        config_dict={"r": 8},
        serialized_named_tensors=ranked_tensors,
    )

    assert captured["endpoint"] == "load_lora_adapter_from_tensors"
    assert captured["payload"]["serialized_tensors"] == ranked_tensors
    assert "serialized_named_tensors" not in captured["payload"]


def test_make_request_logs_error_response(monkeypatch, caplog):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    class ErrorResponse:
        status_code = 400
        text = '{"success":false,"error_message":"invalid adapter tensors"}'

        def raise_for_status(self):
            raise requests.HTTPError("400 Client Error")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: ErrorResponse())

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000

    with caplog.at_level(logging.ERROR), pytest.raises(requests.HTTPError):
        engine._make_request("load_lora_adapter_from_tensors")

    assert "invalid adapter tensors" in caplog.text


def test_init_normal_normalizes_host_before_server_args_resolution(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils import sglang_engine

    captured = {}

    class ReadOnlyServerArgs:
        def __init__(self, **kwargs):
            self.host = kwargs["host"]

    def fake_launch(server_args):
        captured["host"] = server_args.host
        return object()

    monkeypatch.setattr(sglang_engine, "ServerArgs", ReadOnlyServerArgs)
    monkeypatch.setattr(sglang_engine, "launch_server_process", fake_launch)

    engine = sglang_engine.SGLangEngine.__new__(sglang_engine.SGLangEngine)
    engine.node_rank = 1
    engine.router_ip = None
    engine.router_port = None
    server_args_dict = {"host": "[::1]"}

    engine._init_normal(server_args_dict)

    assert captured["host"] == "::1"
    assert server_args_dict["host"] == "[::1]"


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )
