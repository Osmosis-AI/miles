from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.sglang_rollout import generate as gen


async def generate(x):
    sp = x.sampling_params | {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
    }
    return GenerateFnOutput(await gen(x.args, x.sample, sp))
