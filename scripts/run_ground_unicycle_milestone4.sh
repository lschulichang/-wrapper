#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="ground_unicycle_milestone4_old_union"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"
MAX_SPEED_MPS="0.20"
MAX_ACCEL_MPS2="0.30"
MAX_BRAKE_DECEL_MPS2="0.30"
GRID_RASTERIZATION="${GRID_RASTERIZATION:-ellipse_distance}"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"
cat > "$RUN_DIR/cmd.txt" <<EOF
python $ROOT/scripts/smoke_ground_corridor_2d.py --scene old_union --config $CONFIG --z-floor-scene -0.15 --robot-height-meters 0.10 --footprint-radius-meters 0.15 --ground-clearance-meters 0.02 --grid-rasterization $GRID_RASTERIZATION --max-speed-mps $MAX_SPEED_MPS --max-brake-decel-mps2 $MAX_BRAKE_DECEL_MPS2 --max-segment-meters 1.0 --start 0.7209999561 0.2604999840 --goal -0.5350000262 0.2300000042 --output-dir $RUN_DIR/assets
python $ROOT/scripts/smoke_ground_unicycle.py --assets-dir $RUN_DIR/assets --ds-meters 0.02 --max-speed-mps $MAX_SPEED_MPS --max-accel-mps2 $MAX_ACCEL_MPS2 --max-omega-radps 1.0 --max-alpha-radps2 2.0 --dt 0.02 --settle-timeout 5.0 --kx 1.0 --ky 2.0 --ktheta 2.0
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
  --grid-rasterization "$GRID_RASTERIZATION" \
  --max-speed-mps "$MAX_SPEED_MPS" --max-brake-decel-mps2 "$MAX_BRAKE_DECEL_MPS2" \
  --max-segment-meters 1.0 \
  --start 0.7209999561 0.2604999840 --goal -0.5350000262 0.2300000042 \
  --output-dir "$RUN_DIR/assets" |& tee "$RUN_DIR/stdout.log"
python "$ROOT/scripts/smoke_ground_unicycle.py" \
  --assets-dir "$RUN_DIR/assets" \
  --ds-meters 0.02 --max-speed-mps "$MAX_SPEED_MPS" --max-accel-mps2 "$MAX_ACCEL_MPS2" \
  --max-omega-radps 1.0 --max-alpha-radps2 2.0 --dt 0.02 \
  --settle-timeout 5.0 --kx 1.0 --ky 2.0 --ktheta 2.0 |& tee -a "$RUN_DIR/stdout.log"
cat > "$RUN_DIR/notes.md" <<EOF
# Notes

- Goal: time-parameterize the milestone-2 Bezier path and track it with a bounded unicycle/Kanayama controller.
- Planning: direct 2D Dijkstra, projected Gaussian corridor, and degree-6 2D Bezier QP.
- Primary search-map rasterization: $GRID_RASTERIZATION. The milestone-2 stage also records the paired map/Dijkstra comparison for ellipse-distance and AABB-dilation grids.
- Collision-set box: stopping distance is computed from the same 20 cm/s speed limit and a 30 cm/s^2 braking deceleration; the resulting stopping distance is 6.67 cm.
- Tracking defaults: 20 cm/s, 30 cm/s^2, 1 rad/s, 2 rad/s^2, 50 Hz.
- Pipeline: milestone 2 is run first in this same directory; milestone 4 consumes its control points, projected Gaussians, polygons, and grid without rebuilding an independent map.
- Authoritative safety: both the timed reference and every executed segment are checked against the continuous projected-ellipse model.
- Diagnostic: executed-trajectory polygon containment is reported separately because the milestone-2 corridor constrains the reference, not a tracking-error tube.
- Result: see assets/result.json, assets/tracking_result.json, and the six PNG figures.
EOF
echo "run_dir=$RUN_DIR"
