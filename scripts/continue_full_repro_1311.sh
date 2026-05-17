#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/howard/Splat-Nav
RUN_DIR=/home/howard/Splat-Nav/runs/2026-04-23_1311_full_repro_v1
PLAN_CONFIG=/data/howard/splatnav/datasets/splatnav_min/splatnav11202024_images2/config_images2.yml
METHODS=sfc-1,sfc-2,sfc-3,sfc-4,splatplan
TRIALS=180
SEED=20260423
KEEP_TRAJ_LIMIT=12
CONDA_ROOT=/data/howard/miniconda3

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate splatnav

LOC_FRAMES=$(jq length "$RUN_DIR/loc_est_pose.json")
echo "[resume] run_dir=$RUN_DIR"
echo "[resume] loc_frames=$LOC_FRAMES"

if [[ "$LOC_FRAMES" -lt 10 ]]; then
  echo "[resume] ERROR: too few loc frames: $LOC_FRAMES" >&2
  exit 2
fi

cd "$ROOT"

echo "[resume] step=plan_batch start"
env -u LD_LIBRARY_PATH python scripts/run_official_style_batch.py \
  --config "$PLAN_CONFIG" \
  --scene old_union \
  --methods "$METHODS" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --keep_traj_limit "$KEEP_TRAJ_LIMIT" \
  --output "$RUN_DIR/batch_results.json" |& tee "$RUN_DIR/splatplan_batch_stdout.log"

echo "[resume] step=plan_viz start"
env -u LD_LIBRARY_PATH python scripts/visualize_batch_results.py \
  --input "$RUN_DIR/batch_results.json" \
  --output_dir "$RUN_DIR/figs" |& tee "$RUN_DIR/splatplan_viz_stdout.log"

python - "$RUN_DIR/batch_results.json" "$RUN_DIR/report.md" "$LOC_FRAMES" "$RUN_DIR/loc_stable_meta.json" <<'PY'
import datetime
import json
import sys
from pathlib import Path

batch_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
loc_frames = int(sys.argv[3])
loc_meta_path = Path(sys.argv[4])

batch = json.loads(batch_path.read_text(encoding="utf-8"))

rows = []
for r in batch.get("results", []):
    rows.append({
        "method": r["method"],
        "feasible_rate": float(r["feasible_rate"]),
        "avg_plan_time_sec": float(r["avg_plan_time_sec"]),
        "p95_plan_time_sec": float(r["p95_plan_time_sec"]),
        "trials": int(r["trials"]),
    })

rows_sorted = sorted(rows, key=lambda x: (-x["feasible_rate"], x["avg_plan_time_sec"]))

loc_meta = {}
if loc_meta_path.exists():
    try:
        loc_meta = json.loads(loc_meta_path.read_text(encoding="utf-8"))
    except Exception:
        loc_meta = {}

lines = []
lines.append("# Full Repro Report")
lines.append("")
lines.append(f"- generated_at: {datetime.datetime.now().isoformat()}")
lines.append("- scope: Splat-Loc (stable-runner) + official-style Splat-Plan batch on old_union")
lines.append(f"- splatloc_frames_estimated: {loc_frames}")
if loc_meta:
    lines.append(f"- splatloc_detector: {loc_meta.get('detector', '')}")
    lines.append(f"- splatloc_init_mode: {loc_meta.get('init_mode', '')}")
    lines.append(f"- splatloc_success_count: {loc_meta.get('success_count', '')}")
    lines.append(f"- splatloc_fallback_count: {loc_meta.get('fallback_count', '')}")
lines.append(f"- batch_scene: {batch.get('meta', {}).get('scene', 'old_union')}")
lines.append(f"- batch_trials_per_method: {batch.get('meta', {}).get('trials_per_method', '')}")
lines.append("")
lines.append("## Splat-Plan Metrics")
lines.append("")
lines.append("| method | feasible_rate | avg_plan_time_sec | p95_plan_time_sec | trials |")
lines.append("|---|---:|---:|---:|---:|")
for item in rows_sorted:
    lines.append(
        f"| {item['method']} | {item['feasible_rate']:.3f} | {item['avg_plan_time_sec']:.6f} | {item['p95_plan_time_sec']:.6f} | {item['trials']} |"
    )

lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append("- loc_error_stats: `loc_error_stats.png`")
lines.append("- loc_est_pose: `loc_est_pose.json`")
lines.append("- loc_meta: `loc_stable_meta.json`")
lines.append("- plan_batch_json: `batch_results.json`")
lines.append("- plan_fig_summary: `figs/summary.md`")
lines.append("- plan_fig_feasible_rate: `figs/feasible_rate.png`")
lines.append("- plan_fig_plan_time: `figs/plan_time_boxplot.png`")
lines.append("- plan_fig_traj_overlay: `figs/traj_xy_overlay.png`")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[resume] report={report_path}")
PY

echo "[resume] done=$(date --iso-8601=seconds)"
