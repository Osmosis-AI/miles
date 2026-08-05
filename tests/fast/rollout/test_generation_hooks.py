import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generation_hooks import (
    apply_post_generate_hooks,
    apply_pre_generate_hooks,
    load_generate_hooks,
)
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.async_utils import run
from miles.utils.misc import function_registry
from miles.utils.types import Sample


def _input():
    return GenerateFnInput(
        state=SimpleNamespace(args=SimpleNamespace()),
        sample=Sample(prompt="prompt"),
        sampling_params={},
        evaluation=False,
    )


def test_sync_and_async_hooks_compose_in_order():
    def pre(input):
        input.sampling_params["steps"] = ["pre"]

    async def post(input, output):
        await asyncio.sleep(0)
        output.samples.append("post")

    with (
        function_registry.temporary("test:pre", pre),
        function_registry.temporary("test:post", post),
    ):
        input = run(apply_pre_generate_hooks(load_generate_hooks(["test:pre"]), _input()))
        output = run(
            apply_post_generate_hooks(
                load_generate_hooks(["test:post"]),
                input,
                GenerateFnOutput(samples=[]),
            )
        )

    assert input.sampling_params["steps"] == ["pre"]
    assert output.samples == ["post"]


def _args(custom_generate_function_path="test:generator"):
    return SimpleNamespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        custom_generate_function_path=custom_generate_function_path,
    )


def _pre_hook(input):
    input.sampling_params["hook_ran"] = True


def _post_hook(input, output):
    semaphore = getattr(input.state, "generate_fn_semaphore", None) or input.state.semaphore
    assert not semaphore.locked()
    output.samples.metadata["post_hook_ran"] = True
    output.samples.reward = 1.0


async def _generator(input):
    assert input.sampling_params["hook_ran"]
    input.sample.response = "answer"
    input.sample.response_length = 1
    input.sample.tokens = [1, 2]
    input.sample.status = Sample.Status.COMPLETED
    return GenerateFnOutput(samples=input.sample)


@pytest.mark.asyncio
async def test_refactored_rollout_wraps_custom_generator():
    args = _args()
    state = SimpleNamespace(
        args=args,
        aborted=False,
        generate_fn_semaphore=asyncio.Semaphore(1),
        generate_function=_generator,
        pre_generate_hooks=[_pre_hook],
        post_generate_hooks=[_post_hook],
    )

    sample = await inference_rollout_common.generate_and_rm(
        state,
        Sample(prompt="prompt"),
        sampling_params={"max_new_tokens": 1},
    )

    assert sample.metadata["post_hook_ran"]


@pytest.mark.asyncio
async def test_legacy_rollout_wraps_default_generator(monkeypatch):
    args = _args(custom_generate_function_path=None)

    @contextmanager
    def dp_rank_context():
        yield 0

    async def generate(received_args, sample, sampling_params):
        assert received_args is args
        assert sampling_params["hook_ran"]
        sample.response = "answer"
        sample.response_length = 1
        sample.tokens = [1, 2]
        sample.status = Sample.Status.COMPLETED
        return sample

    state = SimpleNamespace(
        args=args,
        aborted=False,
        semaphore=asyncio.Semaphore(1),
        dp_rank_context=dp_rank_context,
        pre_generate_hooks=[_pre_hook],
        post_generate_hooks=[_post_hook],
    )
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _: state)
    monkeypatch.setattr(sglang_rollout, "generate", generate)

    sample = await sglang_rollout.generate_and_rm(
        args,
        Sample(prompt="prompt"),
        sampling_params={"max_new_tokens": 1},
    )

    assert sample.metadata["post_hook_ran"]
