#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <scene> <config_path> [method] [workdir]"
  echo "Example: $0 old_union /data/howard/splatnav/datasets/splatnav_min/splatnav11202024/splatfacto/2024-11-20_120631/config.yml sfc-1 /data/howard/splatnav/datasets/splatnav_min"
  exit 1
fi

SCENE="$1"
CONFIG_PATH="$2"
METHOD="${3:-sfc-1}"
WORKDIR="${4:-/data/howard/splatnav/datasets/splatnav_min}"

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"

RUN_DIR="$("$ROOT/scripts/create_run_dir.sh" "splatplan_smoke_${SCENE}_${METHOD}_tmux")"
OUT_JSON="$RUN_DIR/splatplan_smoke_output.json"

WINDOW_NAME="smoke_${SCENE}_${METHOD//-/_}"
TOPIC="splatplan_smoke_${SCENE}_${METHOD}_tmux_job"

CMD="source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate splatnav && cd $WORKDIR && python $ROOT/scripts/smoke_splatplan.py --scene $SCENE --method $METHOD --config $CONFIG_PATH --output $OUT_JSON"

"$ROOT/scripts/run_in_tmux.sh" "$WINDOW_NAME" "$TOPIC" "$CMD"

echo "smoke_run_dir=$RUN_DIR"
echo "smoke_output=$OUT_JSON"
