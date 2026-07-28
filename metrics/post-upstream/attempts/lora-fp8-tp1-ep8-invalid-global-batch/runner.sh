#!/usr/bin/env bash
set -uo pipefail

: "${RUN_NAME:?set RUN_NAME}"
: "${TP_SIZE:?set TP_SIZE}"
: "${LAUNCH_SOURCE:?set LAUNCH_SOURCE}"

REPO=/root/andy-miles
OUT="/workspace/andy-miles-update/post-upstream/${RUN_NAME}"
HF_CHECKPOINT=/workspace/ablations/qwen3-5-35b-matrix/models/Qwen3.5-35B-A3B-rand-bf16
DATASET=/workspace/dapo-math-17k/dapo-math-17k.jsonl
PROFILE_PID=
LAYERS='language_model.decoder.layers.*'
TARGET_MODULES="${LAYERS}.self_attention.linear_qkv,${LAYERS}.self_attention.linear_proj,${LAYERS}.mlp.experts.linear_fc1,${LAYERS}.mlp.experts.linear_fc2,${LAYERS}.mlp.shared_experts.linear_fc1,${LAYERS}.mlp.shared_experts.linear_fc2,${LAYERS}.self_attention.in_proj,${LAYERS}.self_attention.out_proj"

cleanup() {
  if [[ -n "${PROFILE_PID}" ]]; then
    kill "${PROFILE_PID}" 2>/dev/null || true
    wait "${PROFILE_PID}" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 sglang >/dev/null 2>&1 || true
  pkill -9 miles >/dev/null 2>&1 || true
}
trap cleanup EXIT

rm -rf "${OUT}"
mkdir -p "${OUT}"
cp "${LAUNCH_SOURCE}" "${OUT}/launch_command.sh"
cp "${REPO}/metrics/post-upstream/${RUN_NAME}/profile.yaml" "${OUT}/profile.yaml"
cp "${REPO}/progress-update/scripts/run_post_upstream_lora_fp8.sh" "${OUT}/runner.sh"

cd "${REPO}"
{
  date -u
  git rev-parse HEAD
  git status --porcelain=v1
  echo "run_name=${RUN_NAME}"
  echo "tensor_model_parallel_size=${TP_SIZE}"
  nvidia-smi
  python3 --version
  python3 -c 'import torch, triton; print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)'
  python3 -c 'import transformer_engine; print("transformer_engine", transformer_engine.__version__)'
} >"${OUT}/runtime_profile.txt" 2>&1

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 2 >"${OUT}/gpu_profile.csv" 2>&1 &
PROFILE_PID=$!

ray stop --force >/dev/null 2>&1 || true
pkill -9 sglang >/dev/null 2>&1 || true
pkill -9 miles >/dev/null 2>&1 || true
ray start \
  --head \
  --node-ip-address 127.0.0.1 \
  --num-gpus 8 \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port 8265

source "${REPO}/scripts/models/qwen3.5-35B-A3B.sh"
RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM:/root/andy-miles",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_TIMEOUT_MS": "36000000",
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
    "NVTE_FUSED_ATTN": "0",
    "NVTE_FLASH_ATTN": "1",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"
  }
}'

PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
)
if (( TP_SIZE > 1 )); then
  PARALLEL_ARGS+=(--sequence-parallel)
fi

set +e
timeout 5400 ray job submit \
  --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${REPO}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --megatron-to-hf-mode bridge \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --target-modules "${TARGET_MODULES}" \
  --experts-shared-outer-loras \
  --no-gradient-accumulation-fusion \
  --lora-base-cpu-backup \
  --sglang-lora-backend triton \
  --sglang-max-lora-rank 32 \
  --prompt-data "${DATASET}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --start-rollout-id 0 \
  --num-rollout 1 \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 2 \
  --rollout-max-response-len 64 \
  --rollout-temperature 1 \
  --global-batch-size 4 \
  --balance-data \
  --skip-eval-before-train \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 1e-4 \
  --observe-training-entropy \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --use-tis \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  "${PARALLEL_ARGS[@]}" \
  --qkv-format bshd \
  --micro-batch-size 1 \
  --max-tokens-per-gpu 4096 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --moe-enable-deepep \
  --moe-token-dispatcher-type flex \
  --transformer-impl transformer_engine \
  --bf16 \
  --fp8-format e4m3 \
  --fp8-recipe blockwise \
  --fp8-frozen-base-store \
  --fp8-frozen-base-per-layer-free \
  --rollout-num-gpus-per-engine 4 \
  --sglang-mem-fraction-static 0.4 \
  --sglang-ep-size 1 \
  --sglang-cuda-graph-backend-prefill disabled \
  --sglang-cuda-graph-backend-decode disabled \
  --sglang-dtype bfloat16 \
  --sglang-max-running-requests 64 \
  --sglang-moe-runner-backend triton \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --update-weight-buffer-size 536870912 \
  --calculate-per-token-loss \
  --use-chunked-tp-logprob-loss \
  --use-fused-tp-logprob-kernel \
  --chunked-tp-logprob-seq-chunk-size 128 \
  2>&1 | tee "${OUT}/train.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "${STATUS}" >"${OUT}/exit_status"
python3 "${REPO}/progress-update/scripts/extract_metrics.py" "${OUT}" || true
exit "${STATUS}"
