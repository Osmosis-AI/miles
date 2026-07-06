#!/bin/bash

# One ablation run of the FP8 x chunked/fused-logprob-kernel matrix on
# Qwen3.5-35B-A3B MoE LoRA (colocate, 1x8 GPUs). Parameterized copy of
# examples/lora/run-qwen3.5-35B-A3B-megatron-moe-lora-fp8.sh -- see that
# script for the rationale behind every topology/SGLang flag.
#
# Env contract (set by scripts/k8s/miles-qwen3-5-35b-ablation-matrix.yaml):
#   RUN_NAME     baseline | fp8 | chunked_kernel | fp8_chunked_kernel
#   HF_CKPT      HF checkpoint dir (rand-bf16 or rand-fp8)
#   USE_FP8      1 to train with block-wise FP8 e4m3, else BF16
#   USE_CHUNKED  1 to enable the chunked TP logprob bypass + fused kernel
#   NUM_ROLLOUT  rollouts per run (default 20)
#   OUT_DIR      per-run output dir (checkpoints under $OUT_DIR/ckpts)
#
# All runs share: canonical LoRA q/k/v/o r32, --observe-training-entropy,
# --entropy-coef 1e-4 (random weights give zero rewards, so the entropy term
# is the only gradient source; identical across runs for comparability).

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

: "${RUN_NAME:?}" "${HF_CKPT:?}" "${OUT_DIR:?}"
USE_FP8=${USE_FP8:-0}
USE_CHUNKED=${USE_CHUNKED:-0}
NUM_ROLLOUT=${NUM_ROLLOUT:-20}
RESPONSE_LEN=${RESPONSE_LEN:-4096}

mkdir -p "${OUT_DIR}/ckpts"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "ABLATION RUN: ${RUN_NAME} (fp8=${USE_FP8} chunked=${USE_CHUNKED} ckpt=${HF_CKPT})"

source "/root/miles/scripts/models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   # Bridge mode loads the base model from --hf-checkpoint via Megatron-Bridge.
   # No --ref-load: torch_dist load crashes on this hybrid-GDN model.
   --hf-checkpoint "${HF_CKPT}"
   --save "${OUT_DIR}/ckpts"
   # Force-save fires at the final rollout whenever save-interval is set.
   --save-interval 9999
)

PRECISION_ARGS=(
   --transformer-impl transformer_engine
   --bf16
)
if [ "${USE_FP8}" = "1" ]; then
   PRECISION_ARGS+=(
      --fp8-format e4m3
      --fp8-recipe blockwise
   )
fi

CHUNK_ARGS=()
if [ "${USE_CHUNKED}" = "1" ]; then
   CHUNK_ARGS=(
      --use-chunked-tp-logprob-loss
      --chunked-tp-logprob-seq-chunk-size 256
      --use-fused-tp-logprob-kernel
   )
fi

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --lora-type canonical_lora
   --target-modules "q_proj,k_proj,v_proj,o_proj"
   --sglang-lora-backend triton
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len "${RESPONSE_LEN}"
   --rollout-temperature 1
   # Length stress test: every response runs to the cap, giving a uniform
   # max-length workload across all ablation runs.
   --custom-generate-function-path miles.rollout.generate_hub.benchmarkers.generate_with_ignore_eos

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 9999
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8000
   --eval-top-p 1
)

PERF_ARGS=(
   # TP2 shards the vocab dimension of the loss-path logits: at 12k response
   # length the grad-entropy path holds ~4 full-vocab fp32 copies for
   # backward (~45 GB at TP1), which OOMed both non-chunked runs. TP2 halves
   # every one of them. EP stays 8 (ETP1, expert-DP=1: ETPxEPxEDP = TPxCPxDP
   # = 8) -- 32 experts/rank, ~8 GB less weight memory than EP4.
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # GDN rejects packed sequences; bshd + static micro batches.
   --qkv-format bshd
   --micro-batch-size 1

   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   # Nonzero so gradients flow through the logprob/entropy backward even with
   # zero rewards (random weights); --observe-training-entropy logs it.
   --entropy-coef 1e-4
   --observe-training-entropy
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
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
   # 2 engines x 4 GPUs: block-FP8 [128,128] requires every sharded dim to be
   # a multiple of 128 (shared-expert 512/worldTP). ep=1 dodges the MoE-LoRA
   # align-kernel IMA on -1 expert ids. See the example launcher for details.
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

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --update-weight-buffer-size 536870912 # 512MB
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"NCCL_TIMEOUT_MS\": \"36000000\",
    \"MILES_SKIP_BASE_SYNC\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${PRECISION_ARGS[@]} \
   ${CHUNK_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
