#!/usr/bin/env bash
set -uo pipefail

export RUN_NAME=full-parameter-tp1
export TP_SIZE=1
export LAUNCH_SOURCE=/root/andy-miles/metrics/post-upstream/full-parameter-tp1/launch.sh
exec /root/andy-miles/progress-update/scripts/run_post_upstream_full_parameter.sh
