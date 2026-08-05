"""Privileged-context OPD as composable generation hooks."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from typing import Any

import torch

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, is_lora_enabled
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.types import Sample

_PRIVATE_CONTEXT_KEY = "privileged_opd.private_context"


def _render_prompt(tokenizer: Any, messages: list[dict[str, Any]], tools: Any, args: Namespace) -> list[int]:
    kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
    if tools is not None:
        kwargs["tools"] = tools
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)


def _teacher_messages(prompt: Any, private_context: str) -> list[dict[str, Any]]:
    if (
        not isinstance(prompt, list)
        or not prompt
        or prompt[-1].get("role") != "user"
        or not isinstance(prompt[-1].get("content"), str)
    ):
        raise ValueError("Privileged OPD expects a message-list prompt ending with a text user message.")

    messages = deepcopy(prompt)
    messages[-1]["content"] += (
        "\n\n<privileged_teacher_context>\n"
        "Use this private correction only to predict the assistant response. "
        "Do not quote or reveal it.\n"
        f"{private_context}\n"
        "</privileged_teacher_context>"
    )
    return messages


def _context_limit(args: Namespace) -> int:
    return min(
        int(value)
        for value in (
            getattr(args, "rollout_max_context_len", None),
            getattr(args, "seq_length", None),
            getattr(args, "max_position_embeddings", None),
        )
        if value
    )


def _samples(output: GenerateFnOutput) -> list[Sample]:
    return output.samples if isinstance(output.samples, list) else [output.samples]


def _clear_private_context(sample: Sample, args: Namespace) -> None:
    sample.metadata.pop(args.opsd_private_context_key, None)
    sample.runtime_metadata.pop(_PRIVATE_CONTEXT_KEY, None)


def _take_private_context(sample: Sample, args: Namespace) -> str:
    context = sample.metadata.pop(args.opsd_private_context_key, None)
    context = context or sample.runtime_metadata.get(_PRIVATE_CONTEXT_KEY)
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"Missing private context in metadata[{args.opsd_private_context_key!r}].")
    context = context.strip()
    sample.runtime_metadata[_PRIVATE_CONTEXT_KEY] = context
    return context


def reserve_teacher_context(input: GenerateFnInput) -> None:
    """Hide private context and cap the student response to fit teacher scoring."""

    sample, args = input.sample, input.args
    if input.evaluation:
        _clear_private_context(sample, args)
        return
    if sample.multimodal_inputs:
        raise ValueError("The default privileged OPD hooks support text-only prompts.")

    private_context = _take_private_context(sample, args)
    teacher_prompt = _render_prompt(
        input.state.tokenizer,
        _teacher_messages(sample.prompt, private_context),
        sample.metadata.get("tools"),
        args,
    )
    response_budget = _context_limit(args) - len(teacher_prompt)
    budget = min(input.sampling_params.get("max_new_tokens", response_budget), response_budget)
    if budget < sample.response_length or budget <= 0:
        raise ValueError("The privileged teacher prompt leaves no room for the full student response.")
    input.sampling_params["max_new_tokens"] = budget


def _teacher_endpoint(args: Namespace) -> tuple[str, bool]:
    rm_url = getattr(args, "rm_url", None)
    if not rm_url or rm_url == "self":
        return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate", True
    return rm_url, False


async def _score_teacher(args: Namespace, teacher_tokens: list[int], prompt_length: int) -> dict[str, Any]:
    payload = {
        "input_ids": teacher_tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": max(0, prompt_length - 1),
    }
    url, use_rollout_model = _teacher_endpoint(args)
    if use_rollout_model and is_lora_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME
    return await post(url, payload)


async def score_with_private_context(
    input: GenerateFnInput,
    output: GenerateFnOutput,
) -> GenerateFnOutput:
    """Score each completed student trace under its private teacher context."""

    args = input.args
    samples = _samples(output)
    if input.evaluation:
        for sample in samples:
            _clear_private_context(sample, args)
        return output

    private_context = input.sample.runtime_metadata[_PRIVATE_CONTEXT_KEY]
    for sample in samples:
        if sample.status == Sample.Status.ABORTED:
            sample.metadata.pop(args.opsd_private_context_key, None)
            sample.runtime_metadata[_PRIVATE_CONTEXT_KEY] = private_context
            continue
        if sample.status == Sample.Status.TRUNCATED:
            raise ValueError("Privileged OPD does not train on truncated traces.")
        if sample.response_length <= 0:
            raise ValueError("Privileged OPD requires a generated response.")

        tools = sample.metadata.get("tools")
        public_prompt = _render_prompt(input.state.tokenizer, sample.prompt, tools, args)
        if public_prompt != sample.tokens[: -sample.response_length]:
            raise ValueError("The generator changed the public prompt tokenization.")

        teacher_prompt = _render_prompt(
            input.state.tokenizer,
            _teacher_messages(sample.prompt, private_context),
            tools,
            args,
        )
        response_tokens = sample.tokens[-sample.response_length :]
        teacher_tokens = teacher_prompt + response_tokens
        if len(teacher_tokens) > _context_limit(args):
            raise ValueError("The privileged teacher sequence exceeds the configured context length.")

        sample.reward = await _score_teacher(args, teacher_tokens, len(teacher_prompt))
        _clear_private_context(sample, args)

    return output


def post_process_rewards(
    _args: Namespace,
    samples: list[Sample],
    **_: Any,
) -> tuple[list[float], list[float]]:
    for sample in samples:
        values = sample.reward["meta_info"]["input_token_logprobs"][-sample.response_length :]
        if [int(item[1]) for item in values] != sample.tokens[-sample.response_length :]:
            raise ValueError("Teacher scoring did not preserve the student response tokens.")
        sample.teacher_log_probs = torch.tensor([item[0] for item in values], dtype=torch.float32)

    zero_rewards = [0.0] * len(samples)
    return zero_rewards, zero_rewards
