#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/howard/Splat-Nav
source /data/howard/miniconda3/etc/profile.d/conda.sh
conda activate splatnav
unset LD_LIBRARY_PATH
python "$ROOT/scripts/launch_viser_corridor_from_eval.py" \
  --config /data/howard/splatnav/datasets/splatnav_min/splatnav11202024/splatfacto/2024-11-20_120631/config.yml \
  --eval_json /home/howard/Splat-Nav/runs/2026-04-23_2010_splatplan_eval_plus_old_union_100/eval_plus/eval_results.json \
  --host 0.0.0.0 \
  --port 8091
