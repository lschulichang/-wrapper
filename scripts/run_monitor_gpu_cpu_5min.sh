#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/howard/Splat-Nav
OUT_DIR="$($ROOT/scripts/create_run_dir.sh gpu_cpu_monitor_5min)"
echo "[monitor] out_dir=$OUT_DIR"
bash "$ROOT/scripts/monitor_gpu_cpu_5min.sh" "$OUT_DIR" 300 5
echo "$OUT_DIR" > "$ROOT/runs/latest_gpu_cpu_monitor_5min"
echo "[monitor] done out_dir=$OUT_DIR"
