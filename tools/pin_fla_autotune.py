"""Make fla's Triton autotune deterministic and desync-free under CP.

Appends an env-gated monkeypatch to the installed fla package's utils.py.
fla kernels decorate with ``@triton.autotune(configs=[...], ...)`` at module
import time, and every fla ops/module file imports from ``fla.utils`` first,
so patching ``triton.autotune`` at the end of ``fla/utils.py`` intercepts
every fla kernel decoration in the process.

Why: under context parallelism, mcore GDN interleaves CP collectives inside
the layer while Triton autotune's do_bench re-executes kernels a
rank-dependent number of times. Ranks desynchronize and park in NCCL forever
(observed: all ranks 100% util at ~130 W flat, caches frozen, no progress).

Approach: keep every config (autotune benching doubles as a *feasibility*
filter — pinning to one fixed config OOMs at kernel launch when that config
needs more local memory than the colocated run has free), but replace the
timing benchmark with a single-launch feasibility sweep:

- each rank launches each config exactly once (uniform launch counts across
  ranks -> no collective desync),
- a config that fails to launch scores infinity (feasibility preserved),
- the reported "time" is the evaluation index, so every rank
  deterministically picks the first feasible config.

The patch only activates when FLA_PIN_AUTOTUNE=1 is set in the environment;
otherwise fla behaves stock.

Idempotent: re-running skips if the current marker is present; an older
patch block (previous marker version) is stripped and replaced.
"""

import argparse
import importlib.util
from pathlib import Path

MARKER = "MILES_FLA_PIN_AUTOTUNE_V5"
OLD_MARKERS = ["# === MILES_FLA_PIN_AUTOTUNE (", "# === MILES_FLA_PIN_AUTOTUNE_V"]

PATCH = '''

# === MILES_FLA_PIN_AUTOTUNE_V5 (appended by miles tools/pin_fla_autotune.py) ===
# Deterministic single-launch autotune when FLA_PIN_AUTOTUNE=1: benchmark
# timing loops re-execute kernels a rank-dependent number of times, which
# desyncs CP ranks that interleave collectives inside the GDN layer. Here
# each config is launched exactly once (feasibility probe); "time" is the
# evaluation index, so every rank picks the first feasible config. Configs
# are pre-sorted strongest-first (descending warps*stages): at the long-T
# shapes these probes run, weak configs are 20-50x slower, and evaluation
# order is the pick order.
#
# Conv kernels (causal_conv1d_*) get NO probe launches at all: they run at
# full sequence length per rank (applied after the CP gather), where at
# ~200k tokens a pathological config runs quasi-infinitely under
# cuda.synchronize — even a single-launch sweep is unbounded (observed:
# 25+ min silent inside causal_conv1d_bwd_kernel autotune). Keeping exactly
# one median-strength config makes triton skip benching entirely.
if os.environ.get("FLA_PIN_AUTOTUNE", "0") == "1":
    _pin_orig_autotune = triton.autotune

    def _pin_config_rank(cfg):
        return -((getattr(cfg, "num_warps", 4) or 4) * (getattr(cfg, "num_stages", 1) or 1))

    def _pin_det_bench(kernel_call, quantiles=None, **_kw):
        _pin_det_bench._idx += 1
        try:
            kernel_call()
            torch.cuda.synchronize()
            val = float(_pin_det_bench._idx)
        except Exception:
            # Launch-infeasible config (e.g. CUDA OOM on local-memory-heavy
            # variants); triton's own catch-list misses launch RuntimeErrors.
            val = float("inf")
        return (val, val, val) if quantiles is not None else val

    _pin_det_bench._idx = 0

    def _pin_autotune(configs=None, key=None, **kwargs):
        kwargs["do_bench"] = _pin_det_bench
        # Persisted "timings" from the index trick would be bogus across runs.
        kwargs.pop("cache_results", None)
        if configs is not None:
            configs = sorted(configs, key=_pin_config_rank)

        def _pin_decorator(fn):
            cfgs = configs
            name = getattr(fn, "__name__", None) or getattr(getattr(fn, "fn", None), "__name__", "")
            if cfgs and "conv" in name:
                # Strongest config (list is pre-sorted descending warps*stages):
                # weak/mid configs measured 20-50x slower at long T — at 200k the
                # median pick ran quasi-infinitely (observed: 50 min frozen).
                cfgs = [cfgs[0]]
            return _pin_orig_autotune(configs=cfgs, key=key, **kwargs)(fn)

        return _pin_decorator

    triton.autotune = _pin_autotune
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fla-path",
        default=None,
        help="Path to the installed fla package (default: discovered via importlib)",
    )
    args = parser.parse_args()

    if args.fla_path:
        utils_py = Path(args.fla_path) / "utils.py"
    else:
        spec = importlib.util.find_spec("fla")
        if spec is None or spec.origin is None:
            raise SystemExit("fla package not found; pass --fla-path")
        utils_py = Path(spec.origin).parent / "utils.py"

    source = utils_py.read_text()
    if MARKER in source:
        print(f"already patched: {utils_py}")
        return

    for old in OLD_MARKERS:
        idx = source.find(old)
        if idx != -1:
            source = source[:idx].rstrip() + "\n"
            print("stripped previous patch block")
            break

    utils_py.write_text(source + PATCH)
    print(f"patched: {utils_py}")


if __name__ == "__main__":
    main()
