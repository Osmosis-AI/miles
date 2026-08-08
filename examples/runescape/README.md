# RuneBench GRPO (fishing POC)

Trains Qwen3.6-35B-A3B (RuneBench SFT checkpoint) on the `fishing-xp-15m`
task with flat GRPO. Fork of `examples/swe-agent`; run through
osmosis-traingate (`specs/dev/job/runescape-grpo.toml` in that repo), not a
local launcher.

Reward: **mean XP rate** (total XP ÷ 15-minute episode), written by the
RuneBench verifier to `reward.txt`. `reward.json` rides back as `eval_report`
carrying `peakXpRate` (the leaderboard metric), `meanXpRate`, and the 15s XP
tracker series.

| File | Purpose |
| --- | --- |
| `swe_agent_function.py` | swe-agent original + `trial_dir` passthrough for artifact logging. |
| `generate.py` | reward hook, `benchmark/peak_xp_rate_*` metrics, MLflow artifact logging (videos + reward reports). |
| `make_prompt_data.py` | builds the single-row training JSONL from a task dir. |

## Wiring (via traingate `[miles.args]`)

```
custom-generate-function-path = "miles.rollout.generate_hub.agentic_tool_call.generate"
custom-agent-function-path    = "examples.runescape.swe_agent_function.run"
custom-rm-path                = "examples.runescape.generate.reward_func"
rollout-function-path         = "examples.runescape.generate.RolloutFn"
use-session-server            = true
tito-model                    = "qwen35"   # CPU-verified append-only for Qwen3.6 (11/11)
```

Agent side: the Harbor agent server (harbor-miles branch) with
`RuneBench/agents/rune_mini_swe_agent.py` on PYTHONPATH; training data sets
`metadata.agent_name = "rune_mini_swe_agent:RuneMiniSweAgent"`. Tasks must be
generated with `RUNEBENCH_CLI_ONLY=1 bun generate-tasks.ts` (bash-only,
mean-rate scoring, no MCP).

## MLflow

Metrics per step: `benchmark/peak_xp_rate_{max,mean}` (leaderboard axis —
old-harness SFT ≥305, GLM 5.2 teacher 945), `reward/mean_xp_rate_{mean,max,std}`,
`benchmark/nonzero_frac`, plus the swe-agent timing metrics.

Artifacts per step under `rollouts/step_<n>/sample_<i>/`: every sample's
`reward.json` (with tracker series), plus `recording.mp4` for the best, median,
and worst episode of the group. `RUNESCAPE_LOG_ALL_VIDEOS=1` logs all videos;
`RUNESCAPE_MLFLOW_ARTIFACTS=0` disables artifact logging. Videos require the
Harbor trial dirs on storage the trainer can read (Weka).
