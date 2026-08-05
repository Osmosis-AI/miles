"""Composable hooks around Miles generation functions."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.misc import load_function


def load_generate_hooks(paths: list[str] | None) -> list[Callable]:
    return [load_function(path) for path in paths or []]


async def apply_pre_generate_hooks(
    hooks: list[Callable],
    input: GenerateFnInput,
) -> GenerateFnInput:
    for hook in hooks:
        result = hook(input)
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            input = result
    return input


async def apply_post_generate_hooks(
    hooks: list[Callable],
    input: GenerateFnInput,
    output: GenerateFnOutput,
) -> GenerateFnOutput:
    for hook in hooks:
        result = hook(input, output)
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            output = result
    return output
