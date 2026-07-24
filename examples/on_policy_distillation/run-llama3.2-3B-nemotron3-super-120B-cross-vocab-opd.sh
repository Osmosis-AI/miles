#!/usr/bin/env bash

# xToken-aligned sampled-logprob OPD on one 8xH200 node:
#   GPUs 0-3: colocated Llama-3.2-3B-Instruct actor and rollout engines
#   GPUs 4-7: NVIDIA Nemotron-3-Super-120B-A12B-BF16 SGLang teacher (TP=4)
#
# The teacher has about 247 GB of BF16 weights. Tensor parallelism spreads
# them across four 140 GB H200s, leaving room for Mamba state, KV cache, and
# graphs. This launcher assumes W&B is already logged in (for example through
# `wandb login`) and never handles an API key itself.

set -euo pipefail
set -x

MILES_ROOT=${MILES_ROOT:-/root/miles}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
SGLANG_PYTHON_ROOT=${SGLANG_PYTHON_ROOT:-}
DATA_PATH=${DATA_PATH:-/data/dapo-math-17k/dapo-math-17k.jsonl}

STUDENT_HF_PATH=${STUDENT_HF_PATH:-/data/Llama-3.2-3B-Instruct}
STUDENT_TORCH_DIST_PATH=${STUDENT_TORCH_DIST_PATH:-/data/Llama-3.2-3B-Instruct_torch_dist}
STUDENT_CKPT_PATH=${STUDENT_CKPT_PATH:-/data/Llama-3.2-3B-Instruct_slime}
TEACHER_HF_PATH=${TEACHER_HF_PATH:-/data/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}

TEACHER_HOST=${TEACHER_HOST:-127.0.0.1}
TEACHER_IP=${TEACHER_IP:-127.0.0.1}
TEACHER_PORT=${TEACHER_PORT:-31002}
TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-4,5,6,7}
TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-4}
TEACHER_MEM_FRACTION_STATIC=${TEACHER_MEM_FRACTION_STATIC:-0.85}
MILES_CUDA_VISIBLE_DEVICES=${MILES_CUDA_VISIBLE_DEVICES:-0,1,2,3}
MILES_NUM_GPUS=${MILES_NUM_GPUS:-4}

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_NODE_IP=${RAY_NODE_IP:-127.0.0.1}
RAY_GCS_PORT=${RAY_GCS_PORT:-31000}
RAY_DASHBOARD_HOST=${RAY_DASHBOARD_HOST:-127.0.0.1}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-31001}
RAY_DASHBOARD_AGENT_PORT=${RAY_DASHBOARD_AGENT_PORT:-31003}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-20000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-29999}
MILES_HOST_IP=${MILES_HOST_IP:-127.0.0.1}
MILES_SAFE_WORKDIR=${MILES_SAFE_WORKDIR:-/root/miles_opd_workdir}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/miles_opd_ray_${UID}_$$}
RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-miles-opd-${UID}-$$}

TEACHER_LOG_FILE=${TEACHER_LOG_FILE:-/tmp/sglang_nemotron3_super_120b_teacher_$$.log}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-/tmp/miles_nemotron3_super_120b_opd_train_$$.log}

teacher_pid=
ray_job_submitted=0
ray_submit_pid=
ray_temp_owned=0

is_demo_ray_command() {
    local command=$1

    case "${command}" in
        *"=${RAY_TEMP_DIR}" | *"=${RAY_TEMP_DIR} "* | *"=${RAY_TEMP_DIR}/"*)
            ;;
        *)
            return 1
            ;;
    esac
    case "${command}" in
        *gcs_server* | *raylet* | *ray/dashboard* | *ray/_private* | *ray/autoscaler/_private/monitor.py* | *runtime_env_agent* | *dashboard_agent*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

collect_demo_ray_pids() {
    local pid command

    while read -r pid command; do
        if [[ "${pid}" =~ ^[0-9]+$ ]] && is_demo_ray_command "${command}"; then
            printf '%s\n' "${pid}"
        fi
    done < <(ps -eo pid=,args=)
}

stop_demo_ray() {
    local attempt
    local -a ray_pids=()
    local -a remaining=()

    mapfile -t ray_pids < <(collect_demo_ray_pids)
    if ((${#ray_pids[@]} == 0)); then
        return
    fi

    kill "${ray_pids[@]}" 2>/dev/null || true
    for ((attempt = 0; attempt < 20; attempt++)); do
        mapfile -t remaining < <(collect_demo_ray_pids)
        if ((${#remaining[@]} == 0)); then
            return
        fi
        sleep 1
    done

    kill -KILL "${remaining[@]}" 2>/dev/null || true
}

cleanup() {
    # Once cleanup starts, finish its bounded, run-scoped escalation even if a
    # terminal or parent sends another signal.
    trap '' INT TERM

    local attempt

    # Avoid tracing process command lines while identifying only this demo's
    # Ray children; unrelated commands may contain credentials or other data.
    set +x
    set +e
    if [[ "${ray_job_submitted}" -eq 1 ]]; then
        timeout 15s ray job stop \
            --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
            "${RAY_JOB_SUBMISSION_ID}" \
            >/dev/null 2>&1 || true
    fi
    if [[ -n "${ray_submit_pid}" ]]; then
        for ((attempt = 0; attempt < 20; attempt++)); do
            if ! kill -0 "${ray_submit_pid}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "${ray_submit_pid}" 2>/dev/null; then
            kill -TERM "${ray_submit_pid}" 2>/dev/null || true
        fi
        for ((attempt = 0; attempt < 5; attempt++)); do
            if ! kill -0 "${ray_submit_pid}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "${ray_submit_pid}" 2>/dev/null; then
            kill -KILL "${ray_submit_pid}" 2>/dev/null || true
        fi
        wait "${ray_submit_pid}" 2>/dev/null || true
    fi
    if [[ "${ray_job_submitted}" -eq 1 ]]; then
        timeout 15s ray job stop \
            --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
            "${RAY_JOB_SUBMISSION_ID}" \
            >/dev/null 2>&1 || true
    fi
    if [[ -n "${teacher_pid}" ]] && kill -0 -- "-${teacher_pid}" 2>/dev/null; then
        kill -TERM -- "-${teacher_pid}"
        for ((attempt = 0; attempt < 20; attempt++)); do
            if ! kill -0 -- "-${teacher_pid}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 -- "-${teacher_pid}" 2>/dev/null; then
            kill -KILL -- "-${teacher_pid}"
        fi
        wait "${teacher_pid}"
    fi
    if [[ "${ray_temp_owned}" -eq 1 ]]; then
        stop_demo_ray
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "${RAY_NODE_IP}" ]]; then
    echo "Unable to determine RAY_NODE_IP; set it explicitly." >&2
    exit 1
fi

if python3 - "${TEACHER_IP}" "${TEACHER_PORT}" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
then
    echo "Teacher port is already in use: ${TEACHER_IP}:${TEACHER_PORT}" >&2
    exit 1
fi
if [[ -z "${SGLANG_PYTHON_ROOT}" ]]; then
    if ! SGLANG_PYTHON_ROOT=$(env PYTHONSAFEPATH=1 python3 - <<'PY'
from pathlib import Path

import sglang

print(Path(sglang.__file__).resolve().parent.parent)
PY
    ); then
        echo "Unable to import SGLang; set SGLANG_PYTHON_ROOT to its python source root." >&2
        exit 1
    fi
fi

if [[ "${MILES_ROOT}" != /* || "${MEGATRON_ROOT}" != /* || "${SGLANG_PYTHON_ROOT}" != /* ||
    "${MILES_SAFE_WORKDIR}" != /* || "${RAY_TEMP_DIR}" != /* ]]; then
    echo "MILES_ROOT, MEGATRON_ROOT, SGLANG_PYTHON_ROOT, MILES_SAFE_WORKDIR, and RAY_TEMP_DIR must be absolute paths." >&2
    exit 1
fi

miles_root_real=$(realpath -m "${MILES_ROOT}")
megatron_root_real=$(realpath -m "${MEGATRON_ROOT}")
sglang_python_root_real=$(realpath -m "${SGLANG_PYTHON_ROOT}")
safe_workdir_real=$(realpath -m "${MILES_SAFE_WORKDIR}")
ray_temp_real=$(realpath -m "${RAY_TEMP_DIR}")
if [[ "${safe_workdir_real}" == "${miles_root_real}" || "${safe_workdir_real}" == "${miles_root_real}/"* ]]; then
    echo "MILES_SAFE_WORKDIR must be outside the Miles checkout: ${miles_root_real}" >&2
    exit 1
fi
case "${ray_temp_real}" in
    / | /tmp | /var/tmp | "${miles_root_real}" | "${safe_workdir_real}")
        echo "Refusing unsafe RAY_TEMP_DIR: ${ray_temp_real}" >&2
        exit 1
        ;;
esac
MEGATRON_ROOT="${megatron_root_real}"
SGLANG_PYTHON_ROOT="${sglang_python_root_real}"
RAY_TEMP_DIR="${ray_temp_real}"
if ((${#RAY_TEMP_DIR} > 40)); then
    echo "RAY_TEMP_DIR is too long for Ray's AF_UNIX sockets (maximum 40 characters): ${RAY_TEMP_DIR}" >&2
    exit 1
fi

required_paths=(
    "${MILES_ROOT}/train.py"
    "${MEGATRON_ROOT}/megatron"
    "${SGLANG_PYTHON_ROOT}/sglang/__init__.py"
    "${MILES_ROOT}/scripts/models/llama3.2-3B-Instruct.sh"
    "${DATA_PATH}"
    "${STUDENT_HF_PATH}"
    "${STUDENT_TORCH_DIST_PATH}"
    "${STUDENT_CKPT_PATH}"
    "${TEACHER_HF_PATH}"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

teacher_pythonpath="${SGLANG_PYTHON_ROOT}:${MILES_ROOT}:${MEGATRON_ROOT}"
ray_pythonpath="${MILES_ROOT}:${SGLANG_PYTHON_ROOT}:${MEGATRON_ROOT}"
if ! env PYTHONSAFEPATH=1 PYTHONPATH="${teacher_pythonpath}" python3 - <<'PY'
from sglang.srt.managers.io_struct import BeginWeightUpdateReqInput, EndWeightUpdateReqInput
from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager

assert BeginWeightUpdateReqInput is not None
assert EndWeightUpdateReqInput is not None
assert hasattr(SchedulerWeightUpdaterManager, "begin_weight_update")
assert hasattr(SchedulerWeightUpdaterManager, "end_weight_update")
PY
then
    echo "SGLANG_PYTHON_ROOT must provide the sglang-miles transactional weight-update API." >&2
    exit 1
fi

if [[ -e "${RAY_TEMP_DIR}" ]]; then
    echo "RAY_TEMP_DIR already exists; refusing to reuse another Ray runtime's directory: ${RAY_TEMP_DIR}" >&2
    exit 1
fi
mkdir -p "${MILES_SAFE_WORKDIR}"
mkdir "${RAY_TEMP_DIR}"
ray_temp_owned=1
source "${MILES_ROOT}/scripts/models/llama3.2-3B-Instruct.sh"

CKPT_ARGS=(
    --hf-checkpoint "${STUDENT_HF_PATH}"
    --ref-load "${STUDENT_TORCH_DIST_PATH}"
    --load "${STUDENT_CKPT_PATH}"
    --save "${STUDENT_CKPT_PATH}"
    --save-interval 20
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_PATH}"
    --input-key prompt
    --apply-chat-template
    --opd-prompt-messages-key opd_messages
    --rollout-shuffle
    --num-rollout 300
    --rollout-batch-size 12
    --n-samples-per-prompt 2
    --rollout-max-response-len 2048
    --rollout-temperature 1
    --global-batch-size 24
    --balance-data
)

RM_ARGS=(
    --custom-rm-path miles.rollout.on_policy_distillation.reward_func_cross_vocab
    --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards_cross_vocab
    --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
    --teacher-tokenizer-path "${TEACHER_HF_PATH}"
    --opd-teacher-timeout 600
    --opd-teacher-retries 2
    --opd-teacher-concurrency 8
    --opd-teacher-strict
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type sglang
    --opd-kl-coef 1.0
    --opd-log-prob-top-k 0
    --entropy-coef 0.0
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WANDB_PROJECT=${WANDB_PROJECT:-miles-opd}
WANDB_GROUP=${WANDB_GROUP:-llama3.2-3B-nemotron3-super-120B-cross-vocab-opd}
WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
)

SGLANG_ARGS=(
    --colocate
    --num-gpus-per-node "${MILES_NUM_GPUS}"
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

export MASTER_ADDR
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
# Ray otherwise rewrites an explicit loopback node address to a routable host
# address. This demo is single-node, so keep GCS, agents, and workers local.
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0

cd "${MILES_SAFE_WORKDIR}"

CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" PYTHONPATH="${teacher_pythonpath}" \
    setsid python3 -m sglang.launch_server \
    --model-path "${TEACHER_HF_PATH}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP_SIZE}" \
    --trust-remote-code \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
    >"${TEACHER_LOG_FILE}" 2>&1 &
teacher_pid=$!

teacher_ready=0
for _ in $(seq 1 360); do
    if ! kill -0 "${teacher_pid}" 2>/dev/null; then
        echo "Teacher exited during startup. Log: ${TEACHER_LOG_FILE}" >&2
        tail -n 100 "${TEACHER_LOG_FILE}" >&2
        exit 1
    fi
    if curl -sf --connect-timeout 2 --max-time 5 \
        "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; then
        teacher_ready=1
        break
    fi
    sleep 5
done
if [[ "${teacher_ready}" -ne 1 ]]; then
    echo "Teacher did not become healthy. Log: ${TEACHER_LOG_FILE}" >&2
    tail -n 100 "${TEACHER_LOG_FILE}" >&2
    exit 1
fi
if ! curl -sf --connect-timeout 2 --max-time 5 \
    "http://${TEACHER_IP}:${TEACHER_PORT}/model_info" |
    python3 -c '
import json
import os
import sys

actual = json.load(sys.stdin).get("model_path")
raise SystemExit(os.path.realpath(actual or "") != os.path.realpath(sys.argv[1]))
' "${TEACHER_HF_PATH}"; then
    echo "Teacher model identity does not match ${TEACHER_HF_PATH}. Log: ${TEACHER_LOG_FILE}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${MILES_CUDA_VISIBLE_DEVICES}" ray start \
    --head \
    --node-ip-address "${RAY_NODE_IP}" \
    --port "${RAY_GCS_PORT}" \
    --min-worker-port "${RAY_MIN_WORKER_PORT}" \
    --max-worker-port "${RAY_MAX_WORKER_PORT}" \
    --num-gpus "${MILES_NUM_GPUS}" \
    --disable-usage-stats \
    --include-dashboard=true \
    --dashboard-host="${RAY_DASHBOARD_HOST}" \
    --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_PORT}" \
    --temp-dir "${RAY_TEMP_DIR}"

ray_agent_ready=0
for _ in $(seq 1 60); do
    if curl -sf --connect-timeout 2 --max-time 5 \
        "http://${RAY_NODE_IP}:${RAY_DASHBOARD_AGENT_PORT}/api/healthz" >/dev/null; then
        ray_agent_ready=1
        break
    fi
    sleep 1
done
if [[ "${ray_agent_ready}" -ne 1 ]]; then
    echo "Ray dashboard agent did not become healthy under ${RAY_TEMP_DIR}." >&2
    exit 1
fi

# Keep xtrace disabled for submission so inherited environment credentials can
# never appear in the shell trace. W&B reads the existing local login.
set +x
ray_job_submitted=1
CUDA_VISIBLE_DEVICES="${MILES_CUDA_VISIBLE_DEVICES}" ray job submit \
    --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
    --submission-id="${RAY_JOB_SUBMISSION_ID}" \
    --runtime-env-json='{
        "env_vars": {
            "PYTHONPATH": "'"${ray_pythonpath}"'",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONSAFEPATH": "1",
            "MILES_HOST_IP": "'"${MILES_HOST_IP}"'",
            "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER": "0"
        }
    }' \
    -- python3 "${MILES_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${MILES_NUM_GPUS}" \
    --rollout-num-gpus "${MILES_NUM_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${RM_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG_FILE}" &
ray_submit_pid=$!

teacher_failed=0
teacher_health_failures=0
while kill -0 "${ray_submit_pid}" 2>/dev/null; do
    if ! kill -0 "${teacher_pid}" 2>/dev/null; then
        echo "Teacher exited while the Ray job was running. Log: ${TEACHER_LOG_FILE}" >&2
        teacher_failed=1
    elif curl -sf --connect-timeout 2 --max-time 5 \
        "http://${TEACHER_IP}:${TEACHER_PORT}/health" >/dev/null; then
        teacher_health_failures=0
    else
        ((teacher_health_failures += 1))
        if [[ "${teacher_health_failures}" -ge 6 ]]; then
            echo "Teacher failed six consecutive runtime health checks. Log: ${TEACHER_LOG_FILE}" >&2
            teacher_failed=1
        fi
    fi
    if [[ "${teacher_failed}" -eq 1 ]]; then
        timeout 15s ray job stop \
            --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
            "${RAY_JOB_SUBMISSION_ID}" \
            >/dev/null 2>&1 || true
        break
    fi
    sleep 5
done

if [[ "${teacher_failed}" -eq 1 ]]; then
    exit 1
fi

set +e
wait "${ray_submit_pid}"
ray_job_status=$?
set -e
ray_submit_pid=
ray_job_submitted=0

if [[ "${ray_job_status}" -ne 0 ]]; then
    exit "${ray_job_status}"
fi
