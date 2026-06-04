#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="$1"
DURATION_SEC="${2:-300}"
INTERVAL_SEC="${3:-5}"
mkdir -p "$OUT_DIR"
START_TS=$(date +%s)
END_TS=$((START_TS + DURATION_SEC))

echo "start_iso=$(date --iso-8601=seconds)" > "$OUT_DIR/monitor_meta.txt"
echo "duration_sec=$DURATION_SEC" >> "$OUT_DIR/monitor_meta.txt"
echo "interval_sec=$INTERVAL_SEC" >> "$OUT_DIR/monitor_meta.txt"
echo "host=$(hostname)" >> "$OUT_DIR/monitor_meta.txt"

while [ "$(date +%s)" -lt "$END_TS" ]; do
  NOW=$(date --iso-8601=seconds)
  echo "==== $NOW ====" >> "$OUT_DIR/gpu_apps.log"
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$OUT_DIR/gpu_summary.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >> "$OUT_DIR/gpu_compute_apps.csv" 2>/dev/null || true
  nvidia-smi pmon -c 1 >> "$OUT_DIR/gpu_pmon.log" 2>/dev/null || true
  echo "---- ps top cpu ----" >> "$OUT_DIR/cpu_top.log"
  ps -eo user,pid,ppid,%cpu,%mem,etime,cmd --sort=-%cpu | head -n 25 >> "$OUT_DIR/cpu_top.log"
  echo "---- ps top mem ----" >> "$OUT_DIR/cpu_top.log"
  ps -eo user,pid,ppid,%cpu,%mem,etime,cmd --sort=-%mem | head -n 25 >> "$OUT_DIR/cpu_top.log"
  sleep "$INTERVAL_SEC"
done

echo "end_iso=$(date --iso-8601=seconds)" >> "$OUT_DIR/monitor_meta.txt"
