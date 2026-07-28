#!/usr/bin/env bash
set -uo pipefail

REPO=/root/andy-miles
OUT=/workspace/andy-miles-update/post-upstream/validation-tests

mkdir -p "${OUT}"
cp "$0" "${OUT}/launch.sh"

cd "${REPO}"
{
  date -u
  git rev-parse HEAD
  git status --porcelain=v1
  nvidia-smi
  python3 --version
  python3 -c 'import torch, triton; print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)'
  python3 -c 'import transformer_engine; print("transformer_engine", transformer_engine.__version__)'
} >"${OUT}/runtime_profile.txt" 2>&1

set +e
python3 -m pytest -q \
  tests/fast/backends/megatron_utils/test_lora*.py \
  tests/fast/backends/megatron_utils/test_slice_lora_to_rank.py \
  tests/fast/backends/megatron_utils/test_bridge_lora_helpers.py \
  tests/fast/backends/megatron_utils/test_qwen3_5_mtp_bridge_mapping.py \
  tests/fast/backends/megatron_utils/test_fp8_frozen_base.py \
  tests/fast/backends/megatron_utils/test_chunked_tp_logprob.py \
  tests/fast/backends/megatron_utils/test_selected_tp_logprob_triton.py \
  2>&1 | tee "${OUT}/test.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "${STATUS}" >"${OUT}/exit_status"
PASSED=$(grep -oE '[0-9]+ passed' "${OUT}/test.log" | tail -n 1 | awk '{print $1}')
PASSED=${PASSED:-0}
python3 - "${STATUS}" "${PASSED}" <<'PY' >"${OUT}/metrics.json"
import json
import sys

status = int(sys.argv[1])
passed = int(sys.argv[2])
print(
    json.dumps(
        {
            "status": "passed" if status == 0 else "failed",
            "exit_status": status,
            "passed_tests": passed,
            "expected_tests": 141,
        },
        indent=2,
        sort_keys=True,
    )
)
PY

exit "${STATUS}"
