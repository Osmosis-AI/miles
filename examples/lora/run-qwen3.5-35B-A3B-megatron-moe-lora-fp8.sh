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
   # Bridge mode loads the base model from --hf-checkpoint via Megatron-Bridge.
   # No --ref-load: a torch_dist load routes through Megatron dist_checkpointing,
   # which crashes on this hybrid-GDN model's _extra_state
   # (_replace_sharded_keys_with_state_dict_keys: "BytesIO has no len()"),
   # regardless of which image built the torch_dist. The canonical MoE-LoRA
   # bridge recipe (run-gpt-oss-20B-megatron-moe-lora.sh) loads from HF instead.
   --hf-checkpoint /root/Qwen3.5-35B-A3B-FP8
)

LORA_ARGS=(
   --lora-rank 32                    # LoRA rank
   --lora-alpha 32                   # LoRA alpha (= rank for RL)
   --lora-dropout 0.0                # 0 for RL
   # canonical_lora exports separate q/k/v so SGLang applies them unfused;
   # gated_canonical_lora sizes the gated q adapter to 8192 (query+gate).
   --lora-type canonical_lora
   --target-modules "q_proj,k_proj,v_proj,o_proj"
   --sglang-lora-backend triton
   --megatron-to-hf-mode bridge                      # required for LoRA path
   # base is frozen: keep the SGLang CPU mirror and skip per-step base sync
   --lora-base-cpu-backup
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
   # 4096 avoids the fp32-logits train-step OOM at 8192 on colocated H200s.
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

   # GDN rejects packed sequences; bshd pads per-sequence (needs static micro batches).
   --qkv-format bshd
   --micro-batch-size 1

   # use deepep for megatron MoE
   --moe-enable-deepep
   --moe-token-dispatcher-type flex

   # block-wise FP8
   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
   --fp8-frozen-base-store
   --fp8-frozen-base-per-layer-free
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
   # Block-FP8 needs every sharded dim a multiple of 128. shared_expert=512 and
   # moe_ffn=512 cap the per-engine TP at 4, so use 2 engines x 4 GPUs. ep=1
   # avoids the MoE-LoRA align-kernel IMA on EP's -1 expert sentinels.
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
   --sglang-ep-size 1
   # Hybrid-GDN cuda-graph capture deadlocks under colocate; run eager.
   --sglang-disable-cuda-graph
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
