#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="${TOPIC:-ground_hybrid_fixed_start_compression_ablation_old_union}"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"
TMUX_BIN="$CONDA_ROOT/bin/tmux"
TMUX_LD_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-300}"
SAMPLE_COUNT="${SAMPLE_COUNT:-30}"
SEED="${SEED:-20260718}"

if (( SAMPLE_COUNT <= 0 || SAMPLE_COUNT % 3 != 0 )); then
  echo "SAMPLE_COUNT must be a positive multiple of three" >&2
  exit 2
fi
if (( CANDIDATE_COUNT < SAMPLE_COUNT )); then
  echo "CANDIDATE_COUNT must be at least SAMPLE_COUNT" >&2
  exit 2
fi
PER_LAYER=$((SAMPLE_COUNT / 3))

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"

COMMAND=(
  python "$ROOT/scripts/run_hybrid_compression_ablation.py"
  --scene old_union
  --config "$CONFIG"
  --z-floor-scene -0.15
  --robot-height-meters 0.10
  --footprint-radius-meters 0.15
  --ground-clearance-meters 0.02
  --max-speed-mps 0.20
  --max-brake-decel-mps2 0.30
  --max-segment-meters 1.0
  --endpoint-clearance-meters 0.05
  --min-distance-meters 2.0
  --candidate-count "$CANDIDATE_COUNT"
  --sample-count "$SAMPLE_COUNT"
  --dense-samples 100
  --seed "$SEED"
  --fixed-start 0.7209999561 0.2604999840
  --fixed-start-yaw-rad 3.141592653589793
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
# Fixed-start Hybrid A* coarse-seed compression paired ablation

- The main planner is not changed by this experiment.
- The start pose is fixed at the previously verified old_union pose
  (0.7209999561, 0.2604999840, pi).
- ${CANDIDATE_COUNT} deterministic goal candidates are generated in the fixed
  start's ellipse-distance free component.
- Goals are split into near/medium/far direct-distance terciles. ${PER_LAYER}
  goals per tercile are selected across reference-path turn-count quantiles,
  for ${SAMPLE_COUNT} formal paired trials.
- Goal yaw is unconstrained during Hybrid A*. The downstream Bezier QP
  inherits the terminal yaw selected by Hybrid A*. Start yaw remains fixed
  at pi.
- Each trial runs Hybrid A* once, then reuses the exact same dense coarse path.
- Variant A passes the dense Hybrid A* XY samples directly to corridor construction.
- Variant B splits the pose sequence at every forward/reverse change and applies conservative integer-supercover LOS compression independently inside each motion-direction run.
- The current Hybrid A* implementation is forward-only, so this experiment only observes forward runs; the direction split prevents future reverse-capable planners from shortcutting across a gear cusp.
- Both variants use the same projected-Gaussian map, collision set, corridor builder, endpoint-heading-constrained degree-6 Bezier QP, and alternating within-pair execution order.
- No separate continuous post-hoc circle-ellipse review is included.
EOF

run_job() {
  set +e
  "${COMMAND[@]}" |& tee "$RUN_DIR/stdout.log"
  status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$RUN_DIR/exit_code.txt"
  echo "run_dir=$RUN_DIR" | tee -a "$RUN_DIR/stdout.log"
  return "$status"
}

if [[ "${DETACH:-0}" != "1" ]]; then
  run_job
  exit $?
fi

SESSION="splatnav_jobs"
WINDOW="hybrid_ablation_$(date +%H%M%S)"
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
tmux_cmd new-window -t "$SESSION" -n "$WINDOW" -c "$ROOT" \
  "bash '$JOB_SCRIPT'"
echo "started"
echo "run_dir=$RUN_DIR"
echo "session=$SESSION window=$WINDOW"
