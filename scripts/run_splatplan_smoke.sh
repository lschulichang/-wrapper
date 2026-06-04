#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <scene> <config_path> [method]"
  echo "Example: $0 old_union /home/howard/Splat-Nav/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml sfc-1"
  exit 1
fi

SCENE="$1"
CONFIG_PATH="$2"
METHOD="${3:-sfc-1}"

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
RUN_DIR="$("$ROOT/scripts/create_run_dir.sh" "splatplan_smoke_${SCENE}_${METHOD}")"
OUT_JSON="$RUN_DIR/splatplan_smoke_output.json"

cat > "$RUN_DIR/cmd.txt" << EOF
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate splatnav
python $ROOT/scripts/smoke_splatplan.py --scene $SCENE --method $METHOD --config $CONFIG_PATH --output $OUT_JSON
EOF

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav
unset LD_LIBRARY_PATH

python "$ROOT/scripts/smoke_splatplan.py" \
  --scene "$SCENE" \
  --method "$METHOD" \
  --config "$CONFIG_PATH" \
  --output "$OUT_JSON" |& tee "$RUN_DIR/stdout.log"

echo "run_dir=$RUN_DIR"
