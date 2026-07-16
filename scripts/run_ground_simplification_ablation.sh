#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="ground_simplification_ablation_old_union"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"
mkdir -p "$RUN_DIR/assets"

COMMAND=(
  python "$ROOT/scripts/run_ground_simplification_ablation.py"
  --scene old_union --config "$CONFIG"
  --z-floor-scene -0.15
  --robot-height-meters 0.10
  --footprint-radius-meters 0.15
  --ground-clearance-meters 0.02
  --corridor-margin-meters 0.10
  --max-segment-meters 1.0
  --endpoint-clearance-meters 0.05
  --min-distance-meters 2.0
  --candidate-count 300
  --sample-count 30
  --dense-samples 100
  --seed 20260714
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

cat > "$RUN_DIR/notes.md" <<'EOF'
# Path simplification diagnostic ablation

- A: raw four-connected Dijkstra path, with no simplification.
- B: remove only collinear forward points, with the shared 1 m segment cap.
- C: current sampled inflated-grid LOS simplification, with the shared 1 m cap.
- D: integer supercover inflated-grid LOS simplification, with the shared 1 m cap.
- The same raw Dijkstra path is reused for A/B/C/D for every selected endpoint pair.
- Thirty pairs are sampled from the largest inflated-grid free component and stratified into low/medium/high raw-turn terciles.
- The fixed milestone 2/4 endpoint pair is saved separately and is not included in the paired statistics.
- Primary downstream time excludes map/GS loading and candidate generation.
EOF

"${COMMAND[@]}" |& tee "$RUN_DIR/stdout.log"
echo "run_dir=$RUN_DIR"
