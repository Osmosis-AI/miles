#!/bin/bash
# Benchmark: baseline vs chunked vs fused TP logprob on Qwen3.5-35B-A3B MoE LoRA.
#
# Faithful miles port of slime-rl's scripts/benchmark/bench_chunked_tp_logprob_122b.sh
# (the harness behind osmosis.ai "Cutting Memory in Long-Context RL with Fused
# Logprobs"), adapted to 35B + this cluster's proven SGLang recipe. Reproduces
# the blog's A/B: a materialize-then-chunk baseline (--log-probs-chunk-size)
# vs the hidden-state chunked path, plus the miles-only fused Triton kernel.
#
# Each variant runs NUM_ROLLOUT training steps with identical config; the only
# thing that changes is the logprob/loss path. perf + memory/peak_allocated_gb
# land in the per-variant run.log (see train_metric_utils.log_perf_data_raw).
#
# Env:
#   NUM_ROLLOUT   training steps per variant (default 10; 1 rollout = 1 step)
#   RESPONSE_LEN  rollout-max-response-len (default 12288 -- the blog headline)
#   TP_SIZE       tensor-model-parallel size (default 2, the blog topology)
#   OUT_ROOT      where per-variant run.logs go
set -euo pipefail
set -x

NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
RESPONSE_LEN="${RESPONSE_LEN:-12288}"
TP_SIZE="${TP_SIZE:-2}"
OUT_ROOT="${OUT_ROOT:-/weka/ablations/qwen3-5-35b-logprob-bench}"
USE_FP8="${USE_FP8:-0}"
if [ "${USE_FP8}" = "1" ]; then
    HF_CKPT="${HF_CKPT:-/weka/ablations/qwen3-5-35b-matrix/models/Qwen3.5-35B-A3B-rand-fp8}"
    LABEL_PREFIX="fp8_"
else
    HF_CKPT="${HF_CKPT:-/weka/ablations/qwen3-5-35b-matrix/models/Qwen3.5-35B-A3B-rand-bf16}"
    LABEL_PREFIX=""
fi

mkdir -p "${OUT_ROOT}"

cleanup() {
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
}

export PYTHONBUFFERED=16
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "${NVLINK_COUNT}" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

source "${MODEL_SCRIPT:-/root/miles/scripts/models/qwen3.5-35B-A3B.sh}"

# Sequence parallel is only valid when the vocab is TP-sharded.
SP_ARGS=()
[ "${TP_SIZE}" -gt 1 ] && SP_ARGS=(--sequence-parallel)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --megatron-to-hf-mode bridge
)

# LoRA target is irrelevant to the lm_head logprob path being benchmarked;
# q/k/v/o keeps EP8 clear of the MoE-LoRA align-kernel IMA (bench is bf16).
LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --lora-type canonical_lora
   --target-modules "q_proj,k_proj,v_proj,o_proj"
   --sglang-lora-backend triton
   --megatron-to-hf-mode bridge
)

# Small GBS matching the blog bench (rollout 8 x n-samples 4 = GBS 32).
# ignore_eos forces every response to the cap so the 12K logprob/loss workload
# is uniform across variants (the memory wall the kernel targets).
ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len "${RESPONSE_LEN}"
   --rollout-temperature 1
   --custom-generate-function-path miles.rollout.generate_hub.benchmarkers.generate_with_ignore_eos
   --global-batch-size 32
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 9999
   --eval-prompt-data aime /root/dapo-math-17k/dapo-math-17k.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8000
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --qkv-format bshd
   --micro-batch-size 1
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)

# Random weights => zero rewards, so entropy is the only gradient source and
# must be differentiable through the logprob path (this is what exercises the
# kernel's backward). Identical across all three variants, so the comparison
# stays fair. no ref model => kl-loss-coef 0.
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 1e-4
   --observe-training-entropy
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
   --sglang-ep-size 1
   --sglang-disable-cuda-graph
   --sglang-dtype bfloat16
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 512
   --sglang-moe-runner-backend triton
   --sglang-enable-weights-cpu-backup
)

PRECISION_ARGS=(--transformer-impl transformer_engine --bf16)
[ "${USE_FP8}" = "1" ] && PRECISION_ARGS+=(--fp8-format e4m3 --fp8-recipe blockwise)
[ "${USE_FP8}" = "1" ] && [ "${FP8_PARAM_GATHER:-0}" = "1" ] && PRECISION_ARGS+=(--fp8-param-gather)
[ "${USE_FP8}" = "1" ] && [ "${FP8_FROZEN_BASE:-0}" = "1" ] && PRECISION_ARGS+=(--fp8-frozen-base-store)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --update-weight-buffer-size 536870912
   --calculate-per-token-loss
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_TIMEOUT_MS\": \"36000000\",
    \"MILES_SKIP_BASE_SYNC\": \"1\"
  }
}"

run_variant() {
    local label="$1"; shift
    local logfile="${OUT_ROOT}/${label}.log"
    echo "===== LOGPROB BENCH ${label} (tp=${TP_SIZE} len=${RESPONSE_LEN}) starting $(date -u +%FT%TZ) ====="
    cleanup
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    set +e
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 train.py \
        --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate \
        "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${PRECISION_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" \
        "${PERF_ARGS[@]}" "${SP_ARGS[@]}" "${LORA_ARGS[@]}" "${EVAL_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
        "$@" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "===== LOGPROB BENCH ${label} exited rc=${rc} $(date -u +%FT%TZ) ====="
    cleanup
    sleep 5
    return 0
}

run_variant "${LABEL_PREFIX}baseline" --log-probs-chunk-size 4096
run_variant "${LABEL_PREFIX}chunked512" --use-chunked-tp-logprob-loss --chunked-tp-logprob-seq-chunk-size 512

echo ""
echo "===== RESULTS SUMMARY (fp8=${USE_FP8} tp=${TP_SIZE} len=${RESPONSE_LEN}) ====="
for label in "${LABEL_PREFIX}baseline" "${LABEL_PREFIX}chunked512"; do
    echo "--- ${label} ---"
    grep -aoE "perf [0-9]+: \{.*" "${OUT_ROOT}/${label}.log" 2>/dev/null | tail -1 \
      | grep -aoE "'(perf/log_probs_time|perf/actor_train_time|perf/step_time|memory/peak_allocated_gb|memory/reserved_gb)': [0-9.]+" \
      || grep -aoE "Tried to allocate [0-9.]+ GiB|out of memory" "${OUT_ROOT}/${label}.log" 2>/dev/null | tail -1 \
      || echo "  no perf/OOM line"
done
