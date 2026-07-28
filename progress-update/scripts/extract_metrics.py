#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DICT_RE = re.compile(r"(?:step|perf|rollout) \d+: (\{.*\})")
FP8_RE = re.compile(
    r"fp8 frozen base: quantized (?P<tensors>\d+) tensors .* freed ~(?P<freed_gb>[0-9.]+) GB/rank, "
    r"roundtrip_relerr=(?P<relerr>[0-9.eE+-]+)"
)


def _load_dict(text: str) -> dict[str, Any] | None:
    match = DICT_RE.search(text)
    if match is None:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _gpu_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    by_gpu: dict[str, dict[str, float]] = {}
    samples = 0
    with path.open(newline="", errors="replace") as stream:
        for row in csv.reader(stream):
            if len(row) < 7:
                continue
            samples += 1
            index = row[1].strip()
            try:
                memory_used = float(row[3])
                memory_total = float(row[4])
                utilization = float(row[5])
                power = float(row[6])
            except ValueError:
                continue
            stats = by_gpu.setdefault(
                index,
                {
                    "max_memory_used_mib": 0.0,
                    "memory_total_mib": memory_total,
                    "max_utilization_percent": 0.0,
                    "max_power_watts": 0.0,
                },
            )
            stats["max_memory_used_mib"] = max(stats["max_memory_used_mib"], memory_used)
            stats["max_utilization_percent"] = max(stats["max_utilization_percent"], utilization)
            stats["max_power_watts"] = max(stats["max_power_watts"], power)

    return {
        "sample_rows": samples,
        "per_gpu": by_gpu,
        "max_memory_used_mib": max(
            (stats["max_memory_used_mib"] for stats in by_gpu.values()),
            default=0.0,
        ),
        "max_utilization_percent": max(
            (stats["max_utilization_percent"] for stats in by_gpu.values()),
            default=0.0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    log_path = run_dir / "train.log"
    raw_log = log_path.read_text(errors="replace") if log_path.exists() else ""
    clean_log = ANSI_RE.sub("", raw_log)

    train_metrics: dict[str, Any] | None = None
    train_perf: dict[str, Any] | None = None
    rollout_metrics: dict[str, Any] | None = None
    rollout_perf: dict[str, Any] | None = None

    for line in clean_log.splitlines():
        values = _load_dict(line)
        if values is None:
            continue
        if "train/step" in values:
            train_metrics = values
        elif "perf/actor_train_time" in values:
            train_perf = values
        elif "rollout/response_lengths" in values:
            rollout_metrics = values
        elif "perf/rollout_time" in values:
            rollout_perf = values

    fp8_match = FP8_RE.search(clean_log)
    fp8_metrics = None
    if fp8_match is not None:
        fp8_metrics = {
            "quantized_tensors": int(fp8_match.group("tensors")),
            "freed_gb_per_rank": float(fp8_match.group("freed_gb")),
            "roundtrip_relative_error": float(fp8_match.group("relerr")),
        }

    exit_status_path = run_dir / "exit_status"
    exit_status = int(exit_status_path.read_text().strip()) if exit_status_path.exists() else None
    finite = _all_finite(train_metrics) and _all_finite(train_perf)
    passed = exit_status == 0 and train_metrics is not None and finite

    result = {
        "status": "passed" if passed else "failed",
        "exit_status": exit_status,
        "finite_training_metrics": finite,
        "feature_evidence": {
            "chunked_tp_logprob_installed": "chunked TP logprob bypass installed" in clean_log,
            "fused_tp_logprob_enabled": "--use-fused-tp-logprob-kernel" in clean_log,
            "fp8_frozen_base_quantized": fp8_match is not None,
            "completed_optimizer_step": train_metrics is not None,
            "cuda_oom": "CUDA out of memory" in clean_log,
            "traceback": "Traceback (most recent call last)" in clean_log,
        },
        "train_metrics": train_metrics,
        "train_performance": train_perf,
        "rollout_metrics": rollout_metrics,
        "rollout_performance": rollout_perf,
        "fp8_frozen_base": fp8_metrics,
        "gpu_profile": _gpu_profile(run_dir / "gpu_profile.csv"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
