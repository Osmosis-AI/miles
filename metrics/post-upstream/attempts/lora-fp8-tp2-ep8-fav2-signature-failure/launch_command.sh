#!/usr/bin/env bash
set -uo pipefail

export RUN_NAME=lora-fp8-tp2-ep8
export TP_SIZE=2
export LAUNCH_SOURCE=/root/andy-miles/metrics/post-upstream/lora-fp8-tp2-ep8/launch.sh
exec /root/andy-miles/progress-update/scripts/run_post_upstream_lora_fp8.sh
