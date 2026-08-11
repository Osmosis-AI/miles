"""Pin fla's Triton autotune grids to a single deterministic config.

Appends an env-gated monkeypatch to the installed fla package's utils.py.
fla kernels decorate with ``@triton.autotune(configs=[...], ...)`` at module
import time, and every fla ops/module file imports from ``fla.utils`` first,
so patching ``triton.autotune`` at the end of ``fla/utils.py`` intercepts
every fla kernel decoration in the process.

Why: under context parallelism, mcore GDN interleaves CP collectives inside
the layer while Triton autotune's do_bench re-executes kernels a
rank-dependent number of times. Ranks desynchronize and park in NCCL forever
(observed: all ranks 100% util at ~130 W flat, caches frozen, no progress).
With a single config Triton's Autotuner never benchmarks, so every rank
takes the identical path.

The patch only activates when FLA_PIN_AUTOTUNE=1 is set in the environment;
otherwise fla behaves stock. Config choice is the most conservative one
(smallest block-size product, then fewest warps/stages), identical on every
rank because fla's config lists are static module-level literals.

Idempotent: re-running skips if the marker is already present.
"""

import argparse
import importlib.util
from pathlib import Path

MARKER = "MILES_FLA_PIN_AUTOTUNE"

PATCH = '''

# === MILES_FLA_PIN_AUTOTUNE (appended by miles tools/pin_fla_autotune.py) ===
# Pin every @triton.autotune to one deterministic config when
# FLA_PIN_AUTOTUNE=1: autotune do_bench re-executes kernels a rank-dependent
# number of times, which desyncs CP ranks that interleave collectives inside
# the GDN layer. One config -> Triton skips benchmarking entirely.
if os.environ.get("FLA_PIN_AUTOTUNE", "0") == "1":
    _pin_orig_autotune = triton.autotune

    def _pin_config_cost(cfg):
        block = 1
        for _v in (getattr(cfg, "kwargs", None) or {}).values():
            if isinstance(_v, int):
                block *= max(_v, 1)
        return (block, getattr(cfg, "num_warps", 4) or 4, getattr(cfg, "num_stages", 1) or 1)

    def _pin_autotune(configs=None, key=None, **kwargs):
        cfg_list = list(configs) if configs is not None else []
        if len(cfg_list) > 1:
            cfg_list = [min(cfg_list, key=_pin_config_cost)]
        return _pin_orig_autotune(configs=cfg_list, key=key, **kwargs)

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

    utils_py.write_text(source + PATCH)
    print(f"patched: {utils_py}")


if __name__ == "__main__":
    main()
