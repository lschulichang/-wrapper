#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
WORKDIR="/data/howard/splatnav/datasets/splatnav_min"
SCENE_NAME="${1:-old_union}"
CONFIG_PATH="${2:-/data/howard/splatnav/datasets/splatnav_min/splatnav11202024_images2/config_images2.yml}"
MAX_HOURS="${3:-12}"
TRIALS_PER_METHOD="${4:-200}"
METHODS="${5:-sfc-1,sfc-2,sfc-3,sfc-4,splatplan}"
KEEP_TRAJ_LIMIT="${6:-8}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[overnight] config not found: $CONFIG_PATH"
  exit 1
fi

MAX_SECONDS=$((MAX_HOURS * 3600))
START_TS="$(date +%s)"
ITER=1

echo "[overnight] scene=$SCENE_NAME"
echo "[overnight] config=$CONFIG_PATH"
echo "[overnight] max_hours=$MAX_HOURS"
echo "[overnight] trials_per_method=$TRIALS_PER_METHOD"
echo "[overnight] methods=$METHODS"
echo "[overnight] keep_traj_limit=$KEEP_TRAJ_LIMIT"
echo "[overnight] start_ts=$START_TS"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav

cd "$WORKDIR"

while true; do
  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  if (( ELAPSED >= MAX_SECONDS )); then
    echo "[overnight] reached max runtime (${ELAPSED}s), stop."
    break
  fi

  RUN_DIR="$("$ROOT/scripts/create_run_dir.sh" "formal_batch_${SCENE_NAME}_iter${ITER}")"
  BATCH_JSON="$RUN_DIR/batch_results.json"
  FIG_DIR="$RUN_DIR/figs"

  echo "[overnight] iter=$ITER run_dir=$RUN_DIR"
  echo "[overnight] iter=$ITER running batch..."

  if env -u LD_LIBRARY_PATH python "$ROOT/scripts/run_official_style_batch.py" \
    --config "$CONFIG_PATH" \
    --scene "$SCENE_NAME" \
    --methods "$METHODS" \
    --trials "$TRIALS_PER_METHOD" \
    --seed "$ITER" \
    --keep_traj_limit "$KEEP_TRAJ_LIMIT" \
    --output "$BATCH_JSON"; then
    echo "[overnight] iter=$ITER batch done"
  else
    echo "[overnight] iter=$ITER batch failed, continue next iter"
    ITER=$((ITER + 1))
    continue
  fi

  echo "[overnight] iter=$ITER generating figures..."
  env -u LD_LIBRARY_PATH python "$ROOT/scripts/visualize_batch_results.py" \
    --input "$BATCH_JSON" \
    --output_dir "$FIG_DIR" || true

  echo "[overnight] iter=$ITER finished"
  ITER=$((ITER + 1))
done

echo "[overnight] completed"
