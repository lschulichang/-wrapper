#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
RUNS_ROOT="$ROOT/runs"

SCENE="${1:-old_union}"
MAX_WAIT_HOURS="${2:-14}"
CHECK_INTERVAL_SEC="${3:-120}"

PATTERN="run_overnight_12h.sh $SCENE /data/howard/splatnav/datasets/splatnav_min/splatnav11202024_images2/config_images2.yml 12 180 sfc-1,sfc-2,sfc-3,sfc-4,splatplan 8"

start_ts="$(date +%s)"
max_wait_sec="$((MAX_WAIT_HOURS * 3600))"

echo "[post] scene=$SCENE"
echo "[post] max_wait_hours=$MAX_WAIT_HOURS"
echo "[post] check_interval_sec=$CHECK_INTERVAL_SEC"
echo "[post] waiting for overnight pattern to disappear..."

while true; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if (( elapsed >= max_wait_sec )); then
    echo "[post] timeout waiting overnight stop"
    break
  fi
  if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "[post] overnight alive elapsed_sec=$elapsed"
    sleep "$CHECK_INTERVAL_SEC"
  else
    echo "[post] overnight appears finished"
    break
  fi
done

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav

REPORT_DIR="$("$ROOT/scripts/create_run_dir.sh" "overnight_report_${SCENE}")"
echo "[post] report_dir=$REPORT_DIR"

env -u LD_LIBRARY_PATH python "$ROOT/scripts/aggregate_batch_runs.py" \
  --runs_root "$RUNS_ROOT" \
  --scene "$SCENE" \
  --output_dir "$REPORT_DIR"

# Keep a stable pointer for quick morning check.
ln -sfn "$REPORT_DIR" "$RUNS_ROOT/latest_overnight_report_${SCENE}"

echo "[post] done"
