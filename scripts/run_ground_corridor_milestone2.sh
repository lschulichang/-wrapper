#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"
RUN_DIR="$(cd "$ROOT" && scripts/create_run_dir.sh ground_corridor_milestone2_old_union)"
ASSETS="$RUN_DIR/assets"

cat > "$RUN_DIR/cmd.txt" <<EOF
python $ROOT/scripts/smoke_ground_corridor.py \\
  --config $CONFIG --scene old_union --z-floor -0.15 \\
  --robot-height-meters 0.10 --footprint-radius-meters 0.15 \\
  --ground-clearance-meters 0.02 --corridor-margin-meters 0.10 \\
  --max-segment-meters 1.0 \\
  --start 0.7209999561 0.2604999840 --goal -0.5350000262 0.2300000042 \\
  --output-dir $ASSETS
EOF

cd "$ROOT"
git status --short > "$RUN_DIR/git_status.txt"
git diff > "$RUN_DIR/git_diff.patch"
git ls-files --others --exclude-standard > "$RUN_DIR/untracked_files.txt"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav
unset LD_LIBRARY_PATH

python "$ROOT/scripts/smoke_ground_corridor.py" \
  --config "$CONFIG" --scene old_union --z-floor -0.15 \
  --robot-height-meters 0.10 --footprint-radius-meters 0.15 \
  --ground-clearance-meters 0.02 --corridor-margin-meters 0.10 \
  --max-segment-meters 1.0 \
  --start 0.7209999561 0.2604999840 --goal -0.5350000262 0.2300000042 \
  --output-dir "$ASSETS" |& tee "$RUN_DIR/stdout.log"

cat > "$RUN_DIR/notes.md" <<EOF
# Notes

- Goal: Milestone 2 GS safe corridor and fixed-height 2D Bezier trajectory.
- Result: See assets/result.json and assets/verification.json.
- Model: 15 cm radius, 10 cm high cylinder represented by its circumscribed sphere.
- Corridor margin: 10 cm, independent of dynamics for this milestone.
- Next: Add unicycle reference generation and closed-loop tracking.
EOF

echo "run_dir=$RUN_DIR"
