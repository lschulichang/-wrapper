#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="${TOPIC:-ground_hybrid_raw_collision_evaluation_120_old_union}"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"
SELECTED_GOALS="${SELECTED_GOALS:-$ROOT/runs/2026-07-20_223132_ground_hybrid_free_goal_yaw_compression_ablation_120_old_union/assets/selected_pairs.json}"
SAMPLE_COUNT="${SAMPLE_COUNT:-120}"
TMUX_BIN="$CONDA_ROOT/bin/tmux"
TMUX_LD_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"
COMMAND=(
  python "$ROOT/scripts/run_hybrid_raw_collision_evaluation.py"
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
  --output-dir "$RUN_DIR/assets"
  "$@"
)

printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/cmd.txt"
printf '\n' >> "$RUN_DIR/cmd.txt"
git status --short > "$RUN_DIR/git_status.txt"
git diff > "$RUN_DIR/git_diff.patch"
git ls-files --others --exclude-standard > "$RUN_DIR/untracked_files.txt"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav
unset LD_LIBRARY_PATH
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "python=$(python --version 2>&1)"
  echo "torch=$(python -c 'import torch; print(torch.__version__)')"
  echo "cuda_available=$(python -c 'import torch; print(torch.cuda.is_available())')"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
} > "$RUN_DIR/env.txt"

cat > "$RUN_DIR/notes.md" <<EOF
# Raw Hybrid A* collision evaluation

- Fixed verified start and the protected 120-goal free-terminal-yaw manifest.
- No line-of-sight compression, safety corridor, or Bezier smoothing.
- Reports planner-internal inflated GroundGrid safety separately from the
  continuous circle-ellipse post-check of the dense piecewise-linear path.
- Search failures remain in the fixed denominator and are not resampled.
EOF

snapshot_sources() {
  mkdir -p "$RUN_DIR/source_snapshot/scripts" "$RUN_DIR/source_snapshot/tests" \
    "$RUN_DIR/source_snapshot/splatnav-official/ground_nav"
  cp "$ROOT/scripts/run_hybrid_raw_collision_evaluation.py" \
    "$ROOT/scripts/run_hybrid_raw_collision_evaluation.sh" \
    "$RUN_DIR/source_snapshot/scripts/"
  cp "$ROOT/tests/test_hybrid_raw_collision_evaluation.py" \
    "$RUN_DIR/source_snapshot/tests/"
  cp "$ROOT/splatnav-official/ground_nav/hybrid_astar.py" \
    "$RUN_DIR/source_snapshot/splatnav-official/ground_nav/"
}

if [[ "${DETACH:-0}" != "1" ]]; then
  set +e
  "${COMMAND[@]}" |& tee "$RUN_DIR/stdout.log"
  status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$RUN_DIR/exit_code.txt"
  snapshot_sources
  echo "run_dir=$RUN_DIR" | tee -a "$RUN_DIR/stdout.log"
  exit "$status"
fi

SESSION="splatnav_jobs"
WINDOW="hybrid_collision_$(date +%H%M%S)"
printf -v COMMAND_STRING '%q ' "${COMMAND[@]}"
JOB_SCRIPT="$RUN_DIR/job.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav
unset LD_LIBRARY_PATH
cd "$ROOT"
set +e
$COMMAND_STRING |& tee "$RUN_DIR/stdout.log"
status=\${PIPESTATUS[0]}
echo "\$status" > "$RUN_DIR/exit_code.txt"
mkdir -p "$RUN_DIR/source_snapshot/scripts" "$RUN_DIR/source_snapshot/tests" "$RUN_DIR/source_snapshot/splatnav-official/ground_nav"
cp "$ROOT/scripts/run_hybrid_raw_collision_evaluation.py" "$ROOT/scripts/run_hybrid_raw_collision_evaluation.sh" "$RUN_DIR/source_snapshot/scripts/"
cp "$ROOT/tests/test_hybrid_raw_collision_evaluation.py" "$RUN_DIR/source_snapshot/tests/"
cp "$ROOT/splatnav-official/ground_nav/hybrid_astar.py" "$RUN_DIR/source_snapshot/splatnav-official/ground_nav/"
echo "run_dir=$RUN_DIR" | tee -a "$RUN_DIR/stdout.log"
exit "\$status"
EOF
chmod +x "$JOB_SCRIPT"

tmux_cmd() {
  LD_LIBRARY_PATH="$TMUX_LD_PATH" "$TMUX_BIN" "$@"
}
if ! tmux_cmd has-session -t "$SESSION" 2>/dev/null; then
  tmux_cmd new-session -d -s "$SESSION" -c "$ROOT"
fi
tmux_cmd new-window -t "$SESSION" -n "$WINDOW" -c "$ROOT" "bash '$JOB_SCRIPT'"
echo "started"
echo "run_dir=$RUN_DIR"
echo "session=$SESSION window=$WINDOW"
