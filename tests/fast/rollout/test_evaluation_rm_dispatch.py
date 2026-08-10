import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

from miles.utils.async_utils import run
from miles.utils.types import Sample


def _args():
    return SimpleNamespace(
        custom_generate_function_path=None,
        group_rm=False,
        mask_offpolicy_in_partial_rollout=False,
        partial_rollout=False,
    )


def test_standard_sglang_rollout_marks_eval_reward_calls(monkeypatch):
    from miles.rollout import sglang_rollout

    class GenerateState:
        def __init__(self):
            self.aborted = False
            self.semaphore = asyncio.Semaphore(1)

        def dp_rank_context(self):
            return nullcontext()

    async def generate(_args, sample, _sampling_params):
        return sample

    received_kwargs = {}

    async def reward(_args, _sample, **kwargs):
        received_kwargs.update(kwargs)
        return 1.0

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: GenerateState())
    monkeypatch.setattr(sglang_rollout, "generate", generate)
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)

    sample = Sample(prompt="prompt")
    result = run(sglang_rollout.generate_and_rm(_args(), sample, {}, evaluation=True))

    assert result.reward == 1.0
    assert received_kwargs == {"evaluation": True}


def test_inference_rollout_marks_eval_reward_calls(monkeypatch):
    from miles.rollout.inference_rollout import inference_rollout_common

    args = _args()

    async def generate(generate_input):
        return SimpleNamespace(samples=generate_input.sample)

    state = SimpleNamespace(
        aborted=False,
        args=args,
        generate_fn_semaphore=asyncio.Semaphore(1),
        generate_function=generate,
    )
    received_kwargs = {}

    async def reward(_args, _sample, **kwargs):
        received_kwargs.update(kwargs)
        return 1.0

    monkeypatch.setattr(inference_rollout_common, "async_rm", reward)

    sample = Sample(prompt="prompt")
    result = run(inference_rollout_common.generate_and_rm(state, sample, {}, evaluation=True))

    assert result.reward == 1.0
    assert received_kwargs == {"evaluation": True}
