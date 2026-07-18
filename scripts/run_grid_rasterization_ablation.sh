#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/howard/Splat-Nav"
CONDA_ROOT="/data/howard/miniconda3"
TOPIC="ground_grid_rasterization_ablation_old_union"
CONFIG="$ROOT/splatnav-official/outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"

cd "$ROOT"
RUN_DIR="$(scripts/create_run_dir.sh "$TOPIC")"
mkdir -p "$RUN_DIR/assets"

COMMAND=(
  python "$ROOT/scripts/run_grid_rasterization_ablation.py"
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
  --candidate-count 300
  --sample-count 30
  --dense-samples 100
  --seed 20260718
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
# Projected-Gaussian grid rasterization paired ablation

- Methods: ellipse-distance conservative rasterization versus projected-ellipse AABB fill plus conservative circular-footprint dilation.
- Both methods use the same height-filtered projected Gaussians, map bounds, resolution, robot radius, and 30 endpoint pairs.
- Endpoints must be free in both maps; candidates come from the largest ellipse-distance free component.
- Three strata contain 10 low-, 10 medium-, and 10 high-complexity cases selected from 300 deterministic candidates.
- Each method independently runs 2D Dijkstra, integer-supercover simplification, the same continuous circle-ellipse corridor builder, and the same degree-6 Bezier QP.
- No standalone complete-seed exact collision pass or dense post-hoc Bezier collision pass is used in the formal runtime.
- Method execution order alternates by pair to reduce timing-order bias.
EOF

"${COMMAND[@]}" |& tee "$RUN_DIR/stdout.log"
echo "run_dir=$RUN_DIR"
