#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="${TOPIC:-ground_hybrid_primitive_corridor_stats_old_union}"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"
SELECTED_GOALS="${SELECTED_GOALS:-$ROOT/runs/2026-07-20_223132_ground_hybrid_free_goal_yaw_compression_ablation_120_old_union/assets/selected_pairs.json}"
SAMPLE_COUNT="${SAMPLE_COUNT:-3}"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"

COMMAND=(
  python "$ROOT/scripts/run_hybrid_corridor_simplification.py"
  --scene old_union
  --config "$CONFIG"
  --selected-goals-json "$SELECTED_GOALS"
  --sample-count "$SAMPLE_COUNT"
  --z-floor-scene -0.15
  --robot-height-meters 0.10
  --footprint-radius-meters 0.15
  --ground-clearance-meters 0.02
  --max-speed-mps 0.20
  --max-brake-decel-mps2 0.30
  --min-turning-radius-meters 0.20
  --hybrid-heading-bins 72
  --hybrid-max-expansions 200000
  --curvature-tolerance-1pm 1e-9
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
mkdir -p "$RUN_DIR/source_snapshot/scripts" \
  "$RUN_DIR/source_snapshot/tests" \
  "$RUN_DIR/source_snapshot/splatnav-official/ground_nav"
cp scripts/run_hybrid_corridor_simplification.py \
  scripts/run_hybrid_corridor_simplification.sh \
  "$RUN_DIR/source_snapshot/scripts/"
cp tests/test_hybrid_astar.py \
  tests/test_primitive_path_simplifier.py \
  tests/test_three_layer_planner.py \
  "$RUN_DIR/source_snapshot/tests/"
cp splatnav-official/ground_nav/hybrid_astar.py \
  splatnav-official/ground_nav/primitive_path_simplifier.py \
  splatnav-official/ground_nav/corridor_2d.py \
  splatnav-official/ground_nav/three_layer_planner.py \
  "$RUN_DIR/source_snapshot/splatnav-official/ground_nav/"
echo "run_dir=$RUN_DIR" | tee -a "$RUN_DIR/stdout.log"
exit "$status"
