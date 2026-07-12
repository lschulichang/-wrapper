#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="ground_grid_milestone1_old_union"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"
RESULT_JSON="$RUN_DIR/assets/ground_grid.json"

cat > "$RUN_DIR/cmd.txt" <<EOF
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate splatnav
unset LD_LIBRARY_PATH
python $ROOT/scripts/smoke_ground_grid.py \\
  --scene old_union \\
  --config $CONFIG \\
  --z-floor -0.15 \\
  --robot-height-meters 0.10 \\
  --footprint-radius-meters 0.15 \\
  --ground-clearance-meters 0.02 \\
  --start 0.7209999561 0.2604999840 \\
  --goal -0.5350000262 0.2300000042 \\
  --output $RESULT_JSON
python $ROOT/scripts/verify_ground_grid_result.py $RESULT_JSON
EOF

git status --short > "$RUN_DIR/git_status.txt"
git diff > "$RUN_DIR/git_diff.patch"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav
unset LD_LIBRARY_PATH

python "$ROOT/scripts/smoke_ground_grid.py" \
  --scene old_union \
  --config "$CONFIG" \
  --z-floor -0.15 \
  --robot-height-meters 0.10 \
  --footprint-radius-meters 0.15 \
  --ground-clearance-meters 0.02 \
  --start 0.7209999561 0.2604999840 \
  --goal -0.5350000262 0.2300000042 \
  --output "$RESULT_JSON" |& tee "$RUN_DIR/stdout.log"

python "$ROOT/scripts/verify_ground_grid_result.py" "$RESULT_JSON" \
  | tee "$RUN_DIR/assets/verification.json" \
  | tee -a "$RUN_DIR/stdout.log"

cat > "$RUN_DIR/notes.md" <<EOF
# Notes

- Goal: Milestone 1 ground-grid projection, XY disk dilation, and 2D path planning on old_union.
- Result: Completed successfully; see assets/ground_grid.json and assets/verification.json.
- Parameters: z_floor=-0.15 scene units, height=0.10 m, footprint radius=0.15 m, ground clearance=0.02 m.
- Issues: None in this rerun.
- Next: Connect the verified 2D seed path to the Splat-Plan corridor pipeline.
EOF

echo "run_dir=$RUN_DIR"
