#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="${TOPIC:-ground_three_layer_mainline_old_union}"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"

COMMAND=(
  python "$ROOT/scripts/run_three_layer_ground_planner.py"
  --scene old_union
  --config "$CONFIG"
  --start-pose 0.7209999561 0.2604999840 3.141592653589793
  --goal-xy -0.5350000262 0.2300000042
  --z-floor-scene -0.15
  --robot-height-meters 0.10
  --footprint-radius-meters 0.15
  --ground-clearance-meters 0.02
  --max-speed-mps 0.20
  --max-brake-decel-mps2 0.30
  --min-turning-radius-meters 0.20
  --max-curvature-rate-1pm2 10.0
  --hybrid-heading-bins 72
  --hybrid-max-expansions 200000
  --optimizer-nodes 32
  --optimizer-max-iterations 500
  --optimizer-max-refinements 2
  --output-dir "$RUN_DIR/assets"
  "$@"
)

printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/cmd.txt"
printf '\n' >> "$RUN_DIR/cmd.txt"
git status --short > "$RUN_DIR/git_status.txt"
git diff > "$RUN_DIR/git_diff.patch"
git ls-files --others --exclude-standard \
  > "$RUN_DIR/untracked_files.txt"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav
unset LD_LIBRARY_PATH

set +e
"${COMMAND[@]}" |& tee "$RUN_DIR/stdout.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "$RUN_DIR/exit_code.txt"
echo "run_dir=$RUN_DIR" | tee -a "$RUN_DIR/stdout.log"
exit "$status"
