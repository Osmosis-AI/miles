from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout import privileged_opd
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import single_turn
from miles.utils.types import Sample


class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=True, **kwargs):
        assert not tokenize and add_generation_prompt
        return "\n".join(f"{message['role']}:{message['content']}" for message in messages) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode())


def _args(**overrides):
    values = {
        "opsd_private_context_key": "opsd_targeted_feedback",
        "apply_chat_template_kwargs": {"reasoning_effort": "max"},
        "rollout_max_context_len": 4096,
        "seq_length": 4096,
        "max_position_embeddings": 4096,
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 30000,
        "rm_url": "self",
        "lora_rank": 8,
        "lora_adapter_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _input(args=None, response_ids=None):
    args = args or _args()
    response_ids = response_ids or [900, 901]
    tokenizer = Tokenizer()
    prompt = [{"role": "user", "content": "Solve the public task."}]
    prompt_ids = tokenizer.encode(
        tokenizer.apply_chat_template(prompt, **args.apply_chat_template_kwargs)
    )
    sample = Sample(
        prompt=prompt,
        tokens=prompt_ids + response_ids,
        response="answer",
        response_length=len(response_ids),
        status=Sample.Status.COMPLETED,
        metadata={
            "task_id": "task-1",
            "opsd_targeted_feedback": "Check the denominator before answering.",
        },
    )
    state = SimpleNamespace(args=args, tokenizer=tokenizer)
    return GenerateFnInput(
        state=state,
        sample=sample,
        sampling_params={"temperature": 1.0, "max_new_tokens": 32},
        evaluation=False,
    )


@pytest.mark.asyncio
async def test_privileged_opd_scores_exact_student_trace(monkeypatch):
    input = _input()
    calls = []

    async def fake_post(url, payload):
        calls.append((url, payload))
        return {
            "meta_info": {
                "input_token_logprobs": [
                    [None, payload["input_ids"][-3]],
                    [-0.3, 900],
                    [-0.4, 901],
                ]
            }
        }

    monkeypatch.setattr(privileged_opd, "post", fake_post)

    privileged_opd.reserve_teacher_context(input)
    assert "opsd_targeted_feedback" not in input.sample.metadata

    output = await privileged_opd.score_with_private_context(
        input,
        GenerateFnOutput(samples=input.sample),
    )
    sample = output.samples

    assert calls[0][0] == "http://127.0.0.1:30000/generate"
    assert calls[0][1]["input_ids"][-2:] == [900, 901]
    assert calls[0][1]["lora_path"] == "miles_lora"
    assert b"Check the denominator" in bytes(calls[0][1]["input_ids"][:-2])
    assert sample.metadata == {"task_id": "task-1"}

    raw_rewards, rewards = privileged_opd.post_process_rewards(input.args, [sample])
    assert raw_rewards == rewards == [0.0]
    assert sample.teacher_log_probs.tolist() == pytest.approx([-0.3, -0.4])


@pytest.mark.asyncio
async def test_pre_hook_reserves_context_and_post_hook_rejects_truncation():
    input = _input()
    teacher_prompt_length = len(
        privileged_opd._render_prompt(
            input.state.tokenizer,
            privileged_opd._teacher_messages(
                input.sample.prompt,
                input.sample.metadata["opsd_targeted_feedback"],
            ),
            None,
            input.args,
        )
    )
    input.args.rollout_max_context_len = teacher_prompt_length + 7
    input.args.seq_length = teacher_prompt_length + 7
    input.args.max_position_embeddings = teacher_prompt_length + 7

    privileged_opd.reserve_teacher_context(input)

    assert input.sampling_params["max_new_tokens"] == 7
    input.sample.status = Sample.Status.TRUNCATED
    with pytest.raises(ValueError, match="truncated"):
        await privileged_opd.score_with_private_context(
            input,
            GenerateFnOutput(samples=input.sample),
        )


@pytest.mark.asyncio
async def test_external_teacher_does_not_receive_rollout_lora(monkeypatch):
    input = _input(_args(rm_url="http://teacher.example/generate"), response_ids=[900])
    requests = []

    async def fake_post(url, payload):
        requests.append((url, payload))
        return {"meta_info": {"input_token_logprobs": [[None, 58], [-0.3, 900]]}}

    monkeypatch.setattr(privileged_opd, "post", fake_post)
    privileged_opd.reserve_teacher_context(input)
    await privileged_opd.score_with_private_context(input, GenerateFnOutput(samples=input.sample))

    assert requests[0][0] == "http://teacher.example/generate"
    assert "lora_path" not in requests[0][1]


@pytest.mark.asyncio
async def test_eval_removes_private_context_without_teacher_scoring(monkeypatch):
    input = _input()
    input = GenerateFnInput(
        state=input.state,
        sample=input.sample,
        sampling_params=input.sampling_params,
        evaluation=True,
    )

    async def unexpected_post(*_args, **_kwargs):
        pytest.fail("evaluation should not call the privileged teacher")

    monkeypatch.setattr(privileged_opd, "post", unexpected_post)
    privileged_opd.reserve_teacher_context(input)
    output = await privileged_opd.score_with_private_context(
        input,
        GenerateFnOutput(samples=input.sample),
    )

    assert "opsd_targeted_feedback" not in output.samples.metadata
    assert output.samples.reward is None


@pytest.mark.asyncio
async def test_single_turn_forwards_tools_template_kwargs_and_lora(monkeypatch):
    class RecordingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    tokenizer = RecordingTokenizer()
    args = SimpleNamespace(
        rollout_max_response_len=8,
        rollout_max_context_len=16,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        apply_chat_template_kwargs={"reasoning_effort": "max"},
        lora_rank=8,
        lora_adapter_path=None,
    )
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    sample = Sample(prompt=[{"role": "user", "content": "Solve it."}], metadata={"tools": tools})
    requests = []

    async def fake_post(url, payload):
        requests.append(payload)
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 3]],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(single_turn, "post", fake_post)
    await single_turn.generate(
        GenerateFnInput(
            state=SimpleNamespace(args=args, tokenizer=tokenizer, processor=None),
            sample=sample,
            sampling_params={"max_new_tokens": 8},
            evaluation=False,
        )
    )

    assert tokenizer.kwargs["tools"] == tools
    assert tokenizer.kwargs["reasoning_effort"] == "max"
    assert requests[0]["lora_path"] == "miles_lora"
