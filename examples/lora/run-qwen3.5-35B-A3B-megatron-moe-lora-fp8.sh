#!/bin/bash

# Qwen3.5-35B-A3B GRPO with block-wise FP8 (e4m3) base + o_proj LoRA, colocate.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

source "/root/miles/scripts/models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   # Bridge-load the base from HF; --ref-load (torch_dist) crashes on this
   # hybrid-GDN model's _extra_state.
   --hf-checkpoint /root/Qwen3.5-35B-A3B-FP8
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   # o_proj is the one unfused/ungated attention projection; q/k/v/MoE targets
   # need the gated qkv-LoRA buffer fix upstream in SGLang.
   --lora-type lora
   --target-modules "o_proj"
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
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8000
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # GDN rejects packed sequences; bshd pads per-sequence (needs static batches).
   --qkv-format bshd
   --micro-batch-size 1

   --moe-enable-deepep
   --moe-token-dispatcher-type flex

   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
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

WANDB_ARGS=(
)

SGLANG_ARGS=(
   # Block-FP8 [128,128] needs every sharded dim a multiple of 128, so the
   # shared-expert MLP caps world-TP at 4 => 2 engines x 4 GPUs.
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
   # ep=1 avoids the -1 non-local-expert sentinels that trip the MoE-LoRA align kernel.
   --sglang-ep-size 1
   --sglang-disable-cuda-graph
   --sglang-dtype bfloat16
   --sglang-max-running-requests 512
   --sglang-moe-runner-backend triton
   # Preserve the FP8 base across the colocate torch_memory_saver release/resume:
   # resume() discards weight content, and MILES_SKIP_BASE_SYNC keeps the base from
   # being re-synced, so without this the base is corrupted (gibberish rollouts).
   --sglang-enable-weights-cpu-backup
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --update-weight-buffer-size 536870912
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
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
