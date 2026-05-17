#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/howard/Splat-Nav
CONDA_ROOT=/data/howard/miniconda3
OFFICIAL=/data/howard/repos/splatnav-official
DATA_SRC=/data/howard/splatnav/datasets/splatnav_min/splatnav11202024
MODEL_SRC=/data/howard/splatnav/datasets/splatnav_min/splatnav11202024/splatfacto/2024-11-20_120631
PLAN_CONFIG=/data/howard/splatnav/datasets/splatnav_min/splatnav11202024_images2/config_images2.yml
METHODS=sfc-1,sfc-2,sfc-3,sfc-4,splatplan
TRIALS=180
SEED=20260423
KEEP_TRAJ_LIMIT=12

RUN_DIR="$($ROOT/scripts/create_run_dir.sh full_repro_v1)"
echo "[full] run_dir=$RUN_DIR"
echo "[full] start=$(date --iso-8601=seconds)"

mkdir -p "$RUN_DIR/figs"

{
  echo "plan_config=$PLAN_CONFIG"
  echo "methods=$METHODS"
  echo "trials=$TRIALS"
  echo "seed=$SEED"
} >> "$RUN_DIR/notes.md"

mkdir -p "$OFFICIAL/data" "$OFFICIAL/outputs/old_union2/splatfacto"
ln -sfn "$DATA_SRC" "$OFFICIAL/data/old_union2"
ln -sfn "$MODEL_SRC" "$OFFICIAL/outputs/old_union2/splatfacto/2024-09-02_151414"
# nerfstudio config uses relative path "splatnav11202024" and "outputs/splatnav11202024"
ln -sfn "$DATA_SRC" "$OFFICIAL/splatnav11202024"
ln -sfn "$DATA_SRC" "$OFFICIAL/outputs/splatnav11202024"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav

# For Splat-Loc + gsplat runtime checks (do not override LD_LIBRARY_PATH)
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=/usr/local/cuda-12.0/bin:$PATH

echo "[full] step=loc start"
(
  cd "$OFFICIAL"
  export MPLBACKEND=Agg
  python run_splatloc.py
) |& tee "$RUN_DIR/splatloc_stdout.log"

cp -f "$OFFICIAL/figures/old_union/test_runs/error_stats.png" "$RUN_DIR/loc_error_stats.png"
cp -f "$OFFICIAL/results/old_union/test_runs/est_pose.json" "$RUN_DIR/loc_est_pose.json"

LOC_FRAMES=$(python - "$RUN_DIR/loc_est_pose.json" <<'PY'
import json,sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
PY
)

echo "[full] step=loc done frames=$LOC_FRAMES"

if [[ "$LOC_FRAMES" -lt 10 ]]; then
  echo "[full] ERROR: Splat-Loc produced too few frames: $LOC_FRAMES" >&2
  exit 2
fi

echo "[full] step=plan_batch start"
(
  cd "$ROOT"
  env -u LD_LIBRARY_PATH python scripts/run_official_style_batch.py \
    --config "$PLAN_CONFIG" \
    --scene old_union \
    --methods "$METHODS" \
    --trials "$TRIALS" \
    --seed "$SEED" \
    --keep_traj_limit "$KEEP_TRAJ_LIMIT" \
    --output "$RUN_DIR/batch_results.json"
) |& tee "$RUN_DIR/splatplan_batch_stdout.log"

(
  cd "$ROOT"
  env -u LD_LIBRARY_PATH python scripts/visualize_batch_results.py \
    --input "$RUN_DIR/batch_results.json" \
    --output_dir "$RUN_DIR/figs"
) |& tee "$RUN_DIR/splatplan_viz_stdout.log"

python - "$RUN_DIR/batch_results.json" "$RUN_DIR/report.md" "$LOC_FRAMES" <<'PY'
import json
import sys
import datetime

batch_path, report_path, loc_frames = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(batch_path, 'r', encoding='utf-8') as f:
    batch = json.load(f)

rows = []
for r in batch.get('results', []):
    rows.append((
        r['method'],
        float(r['feasible_rate']),
        float(r['avg_plan_time_sec']),
        float(r['p95_plan_time_sec']),
        int(r['trials']),
    ))

rows_sorted = sorted(rows, key=lambda x: (-x[1], x[2]))

with open(report_path, 'w', encoding='utf-8') as out:
    out.write('# Full Repro Report\n\n')
    out.write(f"- generated_at: {datetime.datetime.now().isoformat()}\n")
    out.write('- scope: official Splat-Loc + official-style Splat-Plan batch on old_union\n')
    out.write(f"- splatloc_frames_estimated: {loc_frames}\n")
    out.write(f"- batch_scene: {batch.get('meta', {}).get('scene', 'old_union')}\n")
    out.write(f"- batch_trials_per_method: {batch.get('meta', {}).get('trials_per_method', '')}\n\n")

    out.write('## Splat-Plan Metrics\n\n')
    out.write('| method | feasible_rate | avg_plan_time_sec | p95_plan_time_sec | trials |\n')
    out.write('|---|---:|---:|---:|---:|\n')
    for method, feasible_rate, avg_time, p95_time, trials in rows_sorted:
        out.write(f"| {method} | {feasible_rate:.3f} | {avg_time:.6f} | {p95_time:.6f} | {trials} |\n")

    out.write('\n## Artifacts\n\n')
    out.write('- loc_error_stats: `loc_error_stats.png`\n')
    out.write('- loc_est_pose: `loc_est_pose.json`\n')
    out.write('- plan_batch_json: `batch_results.json`\n')
    out.write('- plan_fig_summary: `figs/summary.md`\n')
    out.write('- plan_fig_feasible_rate: `figs/feasible_rate.png`\n')
    out.write('- plan_fig_plan_time: `figs/plan_time_boxplot.png`\n')
    out.write('- plan_fig_traj_overlay: `figs/traj_xy_overlay.png`\n')
PY

echo "[full] done=$(date --iso-8601=seconds)"
echo "[full] report=$RUN_DIR/report.md"
echo "$RUN_DIR" > "$ROOT/runs/latest_full_repro_v1"
