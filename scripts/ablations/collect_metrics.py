"""Collect and gate metrics from the FP8 x chunked-kernel ablation matrix.

Parses each run's run.log (train/perf lines are Python-repr dicts), writes
per-run metrics.csv plus a cross-run summary.json, and asserts the
correctness gates unless --report-only.

Usage:
  python3 scripts/ablations/collect_metrics.py --root /weka/ablations/qwen3-5-35b-matrix \
      --runs baseline fp8 chunked_kernel fp8_chunked_kernel --num-rollout 20 [--report-only]
"""

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path

TRAIN_RE = re.compile(r"step (\d+): ({'train/[^\n]*})")
PERF_RE = re.compile(r"perf (\d+): ({'perf/[^\n]*})")

SIGNALS = {
    "bypass_installed": "chunked TP logprob bypass installed on",
    "fused_enabled": "fused selected-TP-logprob kernel enabled",
    "moe_lora_skip": "Current LoRA backend does not support LoRA on MoE layers",
}

WARMUP_STEPS = 2


def parse_log(log_path: Path) -> tuple[dict[int, dict], dict[int, dict], dict[str, int]]:
    train_steps: dict[int, dict] = {}
    perf_steps: dict[int, dict] = {}
    signals = {name: 0 for name in SIGNALS}
    with log_path.open(errors="replace") as f:
        for line in f:
            if m := TRAIN_RE.search(line):
                train_steps[int(m.group(1))] = ast.literal_eval(m.group(2))
            elif m := PERF_RE.search(line):
                perf_steps[int(m.group(1))] = ast.literal_eval(m.group(2))
            for name, needle in SIGNALS.items():
                if needle in line:
                    signals[name] += 1
    return train_steps, perf_steps, signals


def write_csv(out_path: Path, train_steps: dict[int, dict], perf_steps: dict[int, dict]) -> None:
    keys = sorted({k for d in train_steps.values() for k in d} | {k for d in perf_steps.values() for k in d})
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + keys)
        for step in sorted(set(train_steps) | set(perf_steps)):
            row = {**train_steps.get(step, {}), **perf_steps.get(step, {})}
            writer.writerow([step] + [row.get(k, "") for k in keys])


def steady_mean(steps: dict[int, dict], key: str) -> float | None:
    vals = [d[key] for s, d in steps.items() if s >= WARMUP_STEPS and key in d]
    return sum(vals) / len(vals) if vals else None


def first_val(steps: dict[int, dict], key: str) -> float | None:
    for s in sorted(steps):
        if key in steps[s]:
            return steps[s][key]
    return None


def check_run(name: str, run: dict, num_rollout: int, is_fp8: bool, is_chunked: bool) -> list[str]:
    failures = []
    signals = run["signals"]
    train = run["train"]
    perf = run["perf"]

    if is_chunked:
        if signals["bypass_installed"] < 1:
            failures.append("chunked bypass never installed")
        if signals["fused_enabled"] < 1:
            failures.append("fused kernel never enabled")
    else:
        if signals["bypass_installed"] > 0:
            failures.append("bypass installed on a non-chunked run")
        if signals["fused_enabled"] > 0:
            failures.append("fused kernel enabled on a non-chunked run")
    if signals["moe_lora_skip"] > 0:
        failures.append("MoE LoRA skip warning present")
    if len(perf) < num_rollout:
        failures.append(f"only {len(perf)}/{num_rollout} perf lines (run incomplete)")

    ess = steady_mean(train, "train/ess_ratio")
    if ess is None or not (0.98 <= ess <= 1.02):
        failures.append(f"ess_ratio {ess}")
    ppo_kl = steady_mean(train, "train/ppo_kl")
    if ppo_kl is None or abs(ppo_kl) >= 1e-2:
        failures.append(f"ppo_kl {ppo_kl}")
    abs_diff = steady_mean(train, "train/train_rollout_logprob_abs_diff")
    abs_diff_cap = 0.10 if is_fp8 else 0.03
    if abs_diff is None or abs_diff >= abs_diff_cap:
        failures.append(f"train_rollout_logprob_abs_diff {abs_diff} (cap {abs_diff_cap})")
    entropy0 = first_val(train, "train/entropy_loss")
    if entropy0 is None or not (10.9 <= entropy0 <= 13.9):
        failures.append(f"step-0 entropy_loss {entropy0} (expect ~ln(vocab)=12.4 +/- 1.5)")
    for step, d in train.items():
        for key, val in d.items():
            if isinstance(val, float) and not math.isfinite(val):
                failures.append(f"non-finite {key}={val} at step {step}")
        grad_norm = d.get("train/grad_norm")
        if grad_norm is not None and grad_norm <= 0:
            failures.append(f"grad_norm {grad_norm} at step {step} (entropy-coef should force >0)")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--num-rollout", type=int, default=20)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    runs = {}
    for name in args.runs:
        log_path = args.root / name / "run.log"
        if not log_path.exists():
            runs[name] = {"error": "run.log missing", "signals": {}, "train": {}, "perf": {}}
            continue
        train, perf, signals = parse_log(log_path)
        write_csv(args.root / name / "metrics.csv", train, perf)
        runs[name] = {"train": train, "perf": perf, "signals": signals}

    summary = {"runs": {}, "comparisons": {}, "failures": {}}
    for name, run in runs.items():
        if "error" in run:
            summary["failures"][name] = [run["error"]]
            continue
        is_fp8 = "fp8" in name
        is_chunked = "chunked" in name
        summary["runs"][name] = {
            "signals": run["signals"],
            "train_steps": len(run["train"]),
            "perf_steps": len(run["perf"]),
            "means": {
                key: steady_mean(run["train"], key)
                for key in (
                    "train/loss",
                    "train/ess_ratio",
                    "train/ppo_kl",
                    "train/train_rollout_logprob_abs_diff",
                    "train/entropy_loss",
                    "train/grad_norm",
                )
            }
            | {
                key: steady_mean(run["perf"], key)
                for key in (
                    "perf/step_time",
                    "perf/actor_train_time",
                    "perf/log_probs_time",
                    "perf/actor_train_tflops",
                    "perf/actor_train_tok_per_s",
                )
            },
            "step0_entropy": first_val(run["train"], "train/entropy_loss"),
        }
        summary["failures"][name] = check_run(name, run, args.num_rollout, is_fp8, is_chunked)

    for base, variant in [("baseline", "chunked_kernel"), ("fp8", "fp8_chunked_kernel"), ("baseline", "fp8")]:
        if base not in summary["runs"] or variant not in summary["runs"]:
            continue
        comparison = {}
        e0_base = summary["runs"][base]["step0_entropy"]
        e0_var = summary["runs"][variant]["step0_entropy"]
        if e0_base is not None and e0_var is not None:
            comparison["step0_entropy_delta"] = e0_var - e0_base
        for key in ("perf/step_time", "perf/actor_train_time", "perf/log_probs_time"):
            t_base = summary["runs"][base]["means"].get(key)
            t_var = summary["runs"][variant]["means"].get(key)
            if t_base and t_var:
                comparison[f"{key}_speedup_pct"] = 100.0 * (t_base - t_var) / t_base
        summary["comparisons"][f"{base}_vs_{variant}"] = comparison

    # Same base weights, same compute path family: entropy at step 0 must match
    # across the chunked/non-chunked pair within each precision.
    for pair in ("baseline_vs_chunked_kernel", "fp8_vs_fp8_chunked_kernel"):
        delta = summary["comparisons"].get(pair, {}).get("step0_entropy_delta")
        if delta is not None and abs(delta) >= 1e-2:
            summary["failures"].setdefault(pair, []).append(f"step-0 entropy delta {delta} (gate 1e-2)")

    out_path = args.root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    all_failures = {k: v for k, v in summary["failures"].items() if v}
    if all_failures:
        print(f"\nGATE FAILURES: {json.dumps(all_failures, indent=2)}")
        if not args.report_only:
            raise SystemExit(1)
    else:
        print("\nALL GATES PASSED")


if __name__ == "__main__":
    main()
