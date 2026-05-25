#!/bin/bash
# HF -> Megatron-Bridge conversion-only repro using the same FP8 args as the
# multi-LoRA dev run. This stops before rollout, adapter registration, training,
# or Megatron -> HF weight export.

set -ex

export GPUS_PER_NODE=8

pkill sglang || true
ray stop --force || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source scripts/models/qwen3.5-35B-A3B.sh

ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS_PER_NODE --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 examples/multi_lora/test_hf_to_megatron_conversion.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --calculate-per-token-loss \
   --use-miles-router \
   ${MODEL_ARGS[@]} \
   \
   --hf-checkpoint /root/Qwen3.6-35B-A3B-FP8 \
   --megatron-to-hf-mode bridge \
   \
   --lora-rank 32 \
   --lora-alpha 32 \
   --lora-dropout 0.0 \
   --target-modules "q_proj,k_proj,v_proj,o_proj" \
   --multi-lora-dir "${SCRIPT_DIR}/adapters" \
   --multi-lora-n-adapters 128 \
   --multi-lora-idle-poll-s 5 \
   --sglang-lora-backend triton \
   --sglang-moe-a2a-backend deepep \
   --sglang-moe-runner-backend deep_gemm \
   --sglang-deepep-mode auto \
   --transformer-impl transformer_engine \
   --bf16 \
   --fp8-format e4m3 \
   --fp8-recipe blockwise \
   --offload-train \
   \
   --prompt-data /root/gsm8k/train.parquet \
   --input-key messages \
   --label-key label \
   --apply-chat-template \
   --rollout-shuffle \
   --num-rollout 50 \
   --rollout-batch-size 256 \
   --n-samples-per-prompt 4 \
   --rollout-max-response-len 4096 \
   --rollout-temperature 1 \
   --global-batch-size 1024 \
   --offload-train \
   \
   --save /tmp/multi_lora_dev2_save \
   --save-interval 1 \
   \
   --advantage-estimator grpo \
   --kl-loss-coef 0.00 \
   --kl-coef 0.00 \
   --entropy-coef 0.00 \
   --eps-clip 0.2 \
   --eps-clip-high 0.28 \
   \
   --optimizer adam \
   --lr 1e-5 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --expert-tensor-parallel-size 1 \
   --micro-batch-size 2 \
   --max-tokens-per-gpu 670 \
   \
   --rollout-num-gpus-per-engine 4 \
   --sglang-tp-size 1 \
   --sglang-ep-size 4 \
   --sglang-mem-fraction-static 0.85 \
   \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   \
   --use-wandb \
   --wandb-host https://wandb.ai/ \
   --wandb-entity artem-osmosis-osmosis-ai \
   --wandb-project miles-multilora \
   --qkv-format bshd \
   --wandb-group qwen3-4B-dev2-dynamic
