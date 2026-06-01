SGLANG_ENABLE_JIT_DEEPGEMM=1 \
SGLANG_DG_CACHE_DIR=/root/.cache/deep_gemm \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.compile_deep_gemm \
  --model-path /data/Qwen3.5-35B-A3B \
  --tp 2 \
  --ep-size 2 \
  --moe-runner-backend deep_gemm \
  --lora-backend triton \
  --attention-backend fa3 \
  --max-running-requests 1024 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.9 \
  --disable-radix-cache \
  --enable-prefill-delayer \
  --prefill-delayer-max-delay-passes 16 \
  --timeout 3600 \
  2>&1 | tee codex_logs/compile_deep_gemm_qwen35_tp4_ep4.log
