#!/bin/bash
# Run all multi-LoRA diagnostic tests.
#
# Test 1: Forward pass independence (local, no cluster)
# Test 2: Training with per-sample log prob diagnostics (cluster)
# Test 3: Training with double-recompute verification (cluster)
#
# Usage:
#   bash examples/multi_lora/run_tests.sh local     # Test 1 only (no GPU cluster needed)
#   bash examples/multi_lora/run_tests.sh diag       # Test 2: per-sample diagnostics
#   bash examples/multi_lora/run_tests.sh double      # Test 3: double recompute
#   bash examples/multi_lora/run_tests.sh all         # All cluster tests sequentially

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

MODE="${1:-local}"

# ── Test 1: Forward pass independence (local) ────────────────────────────
run_local_test() {
    echo "============================================================"
    echo "TEST 1: Forward pass independence (local)"
    echo "============================================================"
    PYTHONPATH="${REPO_DIR}/../mathew-megatron-bridge/src:${PYTHONPATH:-}" \
        python examples/multi_lora/test_forward_independence.py
}

# ── Cluster setup (shared by Tests 2 & 3) ────────────────────────────────
GPUS_PER_NODE=8

cluster_setup() {
    pkill sglang || true
    ray stop --force || true
    sleep 3

    source scripts/models/qwen3-4B.sh
    ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS_PER_NODE --disable-usage-stats
}

run_training_with_hook() {
    local HOOK_PATH="$1"
    local DESC="$2"
    echo "============================================================"
    echo "$DESC"
    echo "Hook: $HOOK_PATH"
    echo "============================================================"

    ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json='{
         "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1"
         }
       }' \
       -- python3 examples/multi_lora/train_multi_lora.py \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node $GPUS_PER_NODE \
       --colocate \
       --calculate-per-token-loss \
       --use-miles-router \
       ${MODEL_ARGS[@]} \
       \
       --hf-checkpoint /root/Qwen3-4B/ \
       --megatron-to-hf-mode bridge \
       --lora-rank 32 \
       --lora-alpha 32 \
       --lora-dropout 0.0 \
       --target-modules "all-linear" \
       --multi-lora-dir "${SCRIPT_DIR}/adapters" \
       --multi-lora-n-adapters 4 \
       --prompt-data /root/gsm8k/train.parquet \
       --input-key messages \
       --label-key label \
       --apply-chat-template \
       --rollout-shuffle \
       --num-rollout 100 \
       --rollout-batch-size 32 \
       --n-samples-per-prompt 8 \
       --rollout-max-response-len 4096 \
       --rollout-temperature 1 \
       --global-batch-size 256 \
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
       --sequence-parallel \
       --pipeline-model-parallel-size 1 \
       --context-parallel-size 1 \
       --expert-model-parallel-size 1 \
       --expert-tensor-parallel-size 1 \
       --use-dynamic-batch-size \
       --max-tokens-per-gpu 9216 \
       \
       --rollout-num-gpus-per-engine 1 \
       --sglang-mem-fraction-static 0.4 \
       \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --accumulate-allreduce-grads-in-fp32 \
       --attention-softmax-in-fp32 \
       --attention-backend flash \
       \
       --custom-megatron-before-train-step-hook-path "$HOOK_PATH" \
       \
       --use-wandb \
       --wandb-host https://wandb.ai/ \
       --wandb-team osmosis-staging \
       --wandb-project miles-multilora \
       --wandb-group qwen3-4B-test
}

# ── Dispatch ──────────────────────────────────────────────────────────────
case "$MODE" in
  local)
    run_local_test
    ;;
  diag)
    cluster_setup
    run_training_with_hook \
      "examples.multi_lora.diagnose_logprob_diff.before_train_step_hook" \
      "TEST 2: Per-sample log prob alignment diagnostics"
    echo ""
    echo ">>> Look for [logprob_diag] lines in the logs."
    echo ">>> Key things to check:"
    echo ">>>   - Are first3_old and first3_rol close? (alignment check)"
    echo ">>>   - Does max_diff vary by adapter? (cross-contamination check)"
    echo ">>>   - Do weight norms diverge between adapters? (training dynamics)"
    ;;
  double)
    cluster_setup
    run_training_with_hook \
      "examples.multi_lora.test_double_recompute.before_train_step_hook" \
      "TEST 3: Double-recompute verification"
    echo ""
    echo ">>> Look for [double_recompute] lines in the logs."
    echo ">>> Key things to check:"
    echo ">>>   - fresh_vs_stored should be ~0 (model unchanged between passes)"
    echo ">>>   - megatron_vs_sglang shows the actual diff per adapter"
    ;;
  all)
    run_local_test
    echo ""
    cluster_setup
    run_training_with_hook \
      "examples.multi_lora.diagnose_logprob_diff.before_train_step_hook" \
      "TEST 2: Per-sample log prob alignment diagnostics"
    # Note: test 3 would need a fresh cluster start; run separately if needed
    ;;
  *)
    echo "Usage: $0 {local|diag|double|all}"
    exit 1
    ;;
esac
