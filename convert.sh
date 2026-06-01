PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/Qwen3.5-35B-A3B/ \
   --save /data/Qwen3.5-35B-A3B_torch_dust 
