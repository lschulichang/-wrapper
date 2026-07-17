#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="ground_corridor_2d_milestone2_old_union"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"
MAX_SPEED_MPS="0.20"
MAX_BRAKE_DECEL_MPS2="0.30"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"
cat > "$RUN_DIR/cmd.txt" <<EOF
python $ROOT/scripts/smoke_ground_corridor_2d.py --scene old_union --config $CONFIG --z-floor-scene -0.15 --robot-height-meters 0.10 --footprint-radius-meters 0.15 --ground-clearance-meters 0.02 --max-speed-mps $MAX_SPEED_MPS --max-brake-decel-mps2 $MAX_BRAKE_DECEL_MPS2 --max-segment-meters 1.0 --start 0.7209999561 0.2604999840 --goal -0.5350000262 0.2300000042 --output-dir $RUN_DIR/assets
EOF
git status --short > "$RUN_DIR/git_status.txt"
git diff > "$RUN_DIR/git_diff.patch"
git ls-files --others --exclude-standard > "$RUN_DIR/untracked_files.txt"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav
unset LD_LIBRARY_PATH
python "$ROOT/scripts/smoke_ground_corridor_2d.py" \
  --scene old_union --config "$CONFIG" --z-floor-scene -0.15 \
  --robot-height-meters 0.10 --footprint-radius-meters 0.15 --ground-clearance-meters 0.02 \
  --max-speed-mps "$MAX_SPEED_MPS" --max-brake-decel-mps2 "$MAX_BRAKE_DECEL_MPS2" \
  --max-segment-meters 1.0 \
  --start 0.7209999561 0.2604999840 --goal -0.5350000262 0.2300000042 \
  --output-dir "$RUN_DIR/assets" |& tee "$RUN_DIR/stdout.log"
cat > "$RUN_DIR/notes.md" <<EOF
# Notes

- Goal: direct 2D Dijkstra, projected Gaussian ellipses, paper-style 2D separating lines, convex corridor, and 2D Bezier QP.
- Robot: 15 cm projected circle; no circumscribed sphere.
- Collision-set box: stopping distance is computed as vmax^2/(2*max_brake_decel) = 6.67 cm; the unshrunk half-width is robot radius + stopping distance.
- Path simplification uses only the conservative inflated-grid line-of-sight condition; exact circle-ellipse collision is a separate consistency verification.
- Figures: voxel_obstacles.png compares raw/dilated voxels; dijkstra_and_simplified_path.png shows both seed paths; projected_ellipses_corridor_bezier.png shows the continuous planar geometry and optimized trajectory.
- Result: see assets/result.json and assets/verification.json.
EOF
echo "run_dir=$RUN_DIR"
