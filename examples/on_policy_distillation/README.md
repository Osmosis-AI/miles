# On-Policy Distillation Examples

The canonical OPD documentation lives in
[`docs/advanced/on-policy-distillation.md`](../../docs/advanced/on-policy-distillation.md).
Keep the algorithm description, arguments, teacher-mode comparison, and
Rethinking OPD top-k recipe there so we do not maintain two copies.

This directory contains runnable examples:

- `run-qwen3-8B-opd.sh`: SGLang teacher server OPD. This script enables
  Rethinking OPD with `--opd-log-prob-top-k 16`, `--opd-top-k-strategy only-student`,
  and `--opd-reward-weight-mode student_p`.
- `run-qwen3-8B-opd-multi-teacher.sh`: Multi-teacher OPD with per-sample routing.
  Math prompts are scored by a Qwen3-32B teacher and code prompts by a
  Qwen3-Coder-30B-A3B teacher, selected via `--opd-teacher-urls` and a per-row
  `{"metadata": {"opd_teacher": ...}}` tag in the dataset.
- `run-qwen3-8B-opd-megatron.sh`: Megatron-loaded teacher OPD.
- `run-llama3.2-3B-nemotron3-super-120B-cross-vocab-opd.sh`: xToken-aligned
  sampled-logprob OPD with a Llama-3.2-3B student on GPUs 0–3 and a TP=4
  Nemotron-3-Super-120B teacher on GPUs 4–7.

Use `--opd-log-prob-top-k 0` to run the original sampled-token OPD path.
Cross-tokenizer OPD also requires top-k `0`, because it aligns sampled
student/teacher response spans rather than comparing a shared vocabulary.
The two required callbacks are
`miles.rollout.on_policy_distillation.reward_func_cross_vocab` and
`miles.rollout.on_policy_distillation.post_process_rewards_cross_vocab`.

Run the single-node cross-tokenizer example after the `/data` model,
checkpoint, and DAPO dataset paths are available and W&B is already logged in.
Use the Miles image's `sglang-miles` build, or set `SGLANG_PYTHON_ROOT` to the
compatible SGLang checkout's `python` directory:

```bash
export SGLANG_PYTHON_ROOT=/path/to/sglang/python
bash examples/on_policy_distillation/run-llama3.2-3B-nemotron3-super-120B-cross-vocab-opd.sh
```

The launcher uses a safe working directory outside the Miles checkout,
dedicated configurable Ray ports/temp storage, and targeted cleanup scoped to
that Ray temp directory.
