#!/bin/bash
# Multi-LoRA dynamic-lifecycle dev run.
#
# Drives a hard-coded register/deregister schedule via
# train_multi_lora_dynamic.py to exercise the online add/remove path:
#   wait → +dapo → +gsm8k → -dapo → -gsm8k+wait → +gsm8k → +dapo → end
#
# Each productive phase runs 3 rollouts; the final phase runs to
# --num-rollout. With 18 total productive rollouts, the final phase gets 6.

set -ex

export GPUS_PER_NODE=8

pkill sglang || true
ray stop --force || true
pkill miles || true
pkill Megatron || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source scripts/models/qwen3.5-35B-A3B.sh

ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS_PER_NODE --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "SGLANG_ENABLE_SPEC_V2": "1"
     }
   }' \
   -- python3 examples/multi_lora/train_multi_lora_static.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} \
   \
   --hf-checkpoint /data/Qwen3.5-35B-A3B \
   --megatron-to-hf-mode bridge \
   --lora-rank 16 \
   --lora-alpha 16 \
   --lora-dropout 0.1 \
   --target-modules "q_proj,k_proj,v_proj" \
   --multi-lora-dir "${SCRIPT_DIR}/adapters" \
   --multi-lora-n-adapters 128  \
   --sglang-moe-runner-backend deep_gemm \
   --multi-lora-idle-poll-s 5 \
   --sglang-lora-backend triton \
   \
   --prompt-data /root/deepscaler_simple_answers_train.jsonl \
   --log-probs-chunk-size 1024 \
   --input-key messages \
   --label-key label \
   --rm-type deepscaler \
   --apply-chat-template \
   --rollout-shuffle \
   --num-rollout 1000 \
   --rollout-batch-size 256 \
   --n-samples-per-prompt 4 \
   --rollout-max-response-len 8192 \
   --rollout-temperature 0.8 \
   --global-batch-size 1024 \
   --over-sampling-batch-size 512 \
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted_or_truncated \
   \
   --save /tmp/test \
   --save-interval 25 \
   --log-interval 1 \
   \
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 4e-4
   \
   --optimizer adam \
   --lr 1e-5 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --use-tis \
   \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --expert-tensor-parallel-size 1 \
   --micro-batch-size 1 \
   --max-tokens-per-gpu 1000 \
   \
	--sglang-attention-backend fa3 \
	--sglang-enable-lora-overlap-loading \
        --sglang-max-loaded-loras 128 \
	--sglang-max-running-requests 1024 \
	--sglang-chunked-prefill-size 4096 \
        --sglang-enable-mixed-chunk \
	--sglang-cuda-graph-bs 1 2 4 8 138 137 135 134 511 512 1021 1022 1023 1024 $(seq 16 8 256) \
	--sglang-mem-fraction-static 0.9 \
	--rollout-num-gpus-per-engine 4 \
	--sglang-ep-size 4 \
	--moe-token-dispatcher-type flex \
   \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --attention-backend flash \
   \
   --use-wandb \
   --wandb-host https://wandb.ai/ \
   --wandb-entity artem-osmosis-osmosis-ai \
   --wandb-project miles-multilora \
   --qkv-format bshd \
   --wandb-group blogpost_ablation_5_8k_10000ex_tis_gspo_aime_r16
