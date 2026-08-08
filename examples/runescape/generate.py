"""RuneBench: reward, benchmark metrics, and MLflow artifact logging.

The generate function is provided by:
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to swe_agent_function.run
(the copy in this directory, which passes trial_dir through).

Reward: the RuneBench verifier writes mean XP rate to reward.txt (the training
signal) and a full report to reward.json — peakXpRate (the leaderboard metric),
meanXpRate, and the 15s XP tracker series. Harbor returns that report as
``eval_report``, so every sample's metadata carries both statistics.

This module adds two things on top of the swe-agent original:

1. Per-step benchmark metrics (``benchmark/peak_xp_rate_*``): training
   optimises mean XP rate, but the tracker must show the leaderboard number.
   Reference points: old-harness SFT >=305, GLM 5.2 teacher 945.
2. Per-step MLflow artifacts: every sample's reward report + trajectory
   pointer, plus episode videos (best/median/worst of the group by default,
   all 32 with RUNESCAPE_LOG_ALL_VIDEOS=1). Requires the Harbor trial dirs to
   be visible from the trainer (shared storage / Weka); silently skips
   otherwise. Disable entirely with RUNESCAPE_MLFLOW_ARTIFACTS=0.
"""

import json
import logging
import os
import shutil
import statistics
import tempfile
from pathlib import Path

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by the RuneBench verifier during generate()."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


# -- Benchmark metrics --


def _eval_report(sample: Sample) -> dict:
    report = sample.metadata.get("eval_report") if sample.metadata else None
    return report if isinstance(report, dict) else {}


def collect_benchmark_metrics(samples: list[Sample]) -> dict:
    """Per-step XP-rate statistics for the tracker (MLflow/W&B)."""
    peaks = [float(_eval_report(s).get("peakXpRate", 0.0)) for s in samples]
    means = [float(_eval_report(s).get("meanXpRate", 0.0)) for s in samples]
    rewards = [float(s.metadata.get("reward", 0.0)) for s in samples]
    if not rewards:
        return {}

    metrics = {
        "benchmark/peak_xp_rate_max": max(peaks),
        "benchmark/peak_xp_rate_mean": statistics.mean(peaks),
        "benchmark/nonzero_frac": sum(1 for r in rewards if r > 0) / len(rewards),
        "reward/mean_xp_rate_mean": statistics.mean(rewards),
        "reward/mean_xp_rate_max": max(rewards),
    }
    if len(rewards) >= 2:
        metrics["reward/mean_xp_rate_std"] = statistics.stdev(rewards)
    return metrics


# -- Agent metrics (unchanged from the swe-agent example) --


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    return [m.get(key, 0) for m in all_metrics]


def _agg_mean(metrics: dict, all_metrics: list[dict], keys: list[str], prefix: str = "agent/", suffix: str = "_mean"):
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging."""
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if hasattr(s, "metadata") and s.metadata and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}

    for key in ["turns", "tool_calls"]:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    _agg_mean(metrics, all_metrics, ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"])
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")

    values = _collect_values(all_metrics, "total_time")
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


# -- MLflow artifacts --


def _find_video(trial_dir: str) -> Path | None:
    """The sandbox ffmpeg-records every episode to logs/verifier/recording.mp4
    inside the trial dir; rglob keeps this robust to trial-dir layout changes."""
    if not trial_dir:
        return None
    root = Path(trial_dir)
    if not root.is_dir():
        return None
    try:
        return next(root.rglob("recording.mp4"), None)
    except OSError:
        return None


def _video_sample_indices(rewards: list[float]) -> list[int]:
    """Best, median, and worst episode of the group (or all, via env flag)."""
    if os.environ.get("RUNESCAPE_LOG_ALL_VIDEOS") == "1":
        return list(range(len(rewards)))
    order = sorted(range(len(rewards)), key=lambda i: rewards[i])
    return sorted({order[-1], order[len(order) // 2], order[0]})


def log_artifacts_to_mlflow(samples: list[Sample], rollout_id: int, args) -> None:
    if os.environ.get("RUNESCAPE_MLFLOW_ARTIFACTS", "1") == "0":
        return
    if not getattr(args, "use_mlflow", False):
        return

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not importable; skipping artifact logging")
        return

    # Attach to the run the primary rank opened (metrics and artifacts land in
    # the same MLflow run).
    if mlflow.active_run() is None:
        run_id = getattr(args, "mlflow_run_id", None) or os.environ.get("MLFLOW_RUN_ID")
        if run_id is None:
            logger.warning("no MLflow run id available; skipping artifact logging")
            return
        mlflow.start_run(run_id=run_id)

    rewards = [float(s.metadata.get("reward", 0.0)) for s in samples]
    video_indices = set(_video_sample_indices(rewards))

    with tempfile.TemporaryDirectory(prefix="runescape-artifacts-") as tmp:
        tmp_root = Path(tmp)
        for i, sample in enumerate(samples):
            sample_dir = tmp_root / f"sample_{i:02d}"
            sample_dir.mkdir()

            report = dict(_eval_report(sample))
            report["exit_status"] = sample.metadata.get("exit_status", "")
            report["trial_dir"] = sample.metadata.get("trial_dir", "")
            (sample_dir / "reward.json").write_text(json.dumps(report, indent=2))

            if i in video_indices:
                video = _find_video(sample.metadata.get("trial_dir", ""))
                if video is not None:
                    try:
                        shutil.copyfile(video, sample_dir / "recording.mp4")
                    except OSError as e:
                        logger.warning(f"could not copy video for sample {i}: {e}")

        try:
            mlflow.log_artifacts(str(tmp_root), artifact_path=f"rollouts/step_{rollout_id:05d}")
        except Exception as e:
            logger.warning(f"mlflow artifact logging failed for step {rollout_id}: {e}")


# -- Rollout Function --


class RolloutFn(InferenceRolloutFn):
    """Rollout function with benchmark metrics + MLflow artifact logging."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        all_samples = []
        for group in output.samples:
            if isinstance(group, list):
                all_samples.extend(group)
            else:
                all_samples.append(group)

        metrics = output.metrics or {}
        metrics.update(aggregate_agent_metrics(all_samples))
        metrics.update(collect_benchmark_metrics(all_samples))
        output.metrics = metrics

        if metrics:
            logger.info(f"RuneBench metrics for rollout {input.rollout_id}: "
                        f"{ {k: v for k, v in metrics.items() if k.startswith(('benchmark/', 'reward/'))} }")

        try:
            # InferenceRolloutFn keeps args on its GenerateState, not on self.
            log_artifacts_to_mlflow(all_samples, input.rollout_id, self.state.args)
        except Exception as e:
            logger.warning(f"artifact logging failed (non-fatal): {e}")

        return output
