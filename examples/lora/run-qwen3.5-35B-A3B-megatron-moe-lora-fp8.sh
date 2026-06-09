#!/bin/bash

# Qwen3.5-35B-A3B MoE LoRA + block-wise FP8 training (Hopper / Blackwell).
#
# Combines:
#   - LoRA on MoE expert projections (gate_proj, up_proj, down_proj)
#   - SGLang triton LoRA backend (required for MoE LoRA)
#   - Megatron-Bridge HF conversion (required for LoRA path)
#   - Block-wise FP8 e4m3 forward, BF16 backward + master weights
#   - --use-tis for MoE numerical drift compensation
#
# See docs/superpowers/plans/2026-06-08-fp8-moe-lora-02-fp8-moe-lora-bringup.md.

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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3.5-35B-A3B-FP8
   --ref-load      /root/Qwen3.5-35B-A3B_torch_dist
)

LORA_ARGS=(
   --lora-rank 32                    # LoRA rank
   --lora-alpha 32                   # LoRA alpha (= rank for RL)
   --lora-dropout 0.0                # 0 for RL
   --target-modules "gate_proj,up_proj,down_proj"   # MoE expert projections
   --sglang-lora-backend triton                      # required for MoE LoRA
   --megatron-to-hf-mode bridge                      # required for LoRA path
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
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
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

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --no-offload-train

   # use deepep for megatron MoE
   --moe-enable-deepep
   --moe-token-dispatcher-type flex

   # block-wise FP8
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
   --use-tis                          # MoE precision-drift compensation
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
   # --use-wandb
   # --wandb-project miles-fp8-moe-lora
   # --wandb-group qwen3.5-35B-A3B-fp8-moe-lora
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   # Block-wise FP8 layout is [128,128], so every tensor-sharded weight
   # sub-partition must stay a multiple of 128. The shared-expert MLP shards
   # across the engine's dense TP, which miles forces to equal
   # --rollout-num-gpus-per-engine (it overwrites --sglang-tp-size; see
   # backends/sglang_utils/arguments.py:136 -- so that flag is a no-op).
   # With shared_expert_intermediate=512 the gate/up sub-partition (512/worldTP)
   # and the down input (512/worldTP) must each be a multiple of 128 => worldTP
   # <= 4. worldTP=8 gives 64 (the prior failure), so run 2 engines x 4 GPUs.
   # ep=4 == worldTP keeps the 256 routed experts whole (moe_tp = worldTP/ep = 1).
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
   --sglang-ep-size 4
   --sglang-dtype bfloat16

   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 512
   --sglang-moe-runner-backend triton
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --update-weight-buffer-size 536870912 # 512MB
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1 forces fp32 scales in fp8 training,
# matching what sglang serves on the rollout side.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"NCCL_TIMEOUT_MS\": \"36000000\"
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
