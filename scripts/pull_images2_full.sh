#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/data/howard/splatnav/datasets/splatnav_min/splatnav11202024/images2_flat}"
MAX_ATTEMPTS="${2:-30}"

mkdir -p "$OUT_DIR"

for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
  echo "[images2] attempt=${attempt}"
  gdown --folder "https://drive.google.com/drive/folders/1ihEkWkGQ8PPbIMY-fAuKQyDchF_rCLEz" \
    --continue \
    --remaining-ok \
    -O "$OUT_DIR" || true

  count="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d ' ')"
  echo "[images2] count=${count}"

  if [[ "$count" -ge 50 ]]; then
    echo "[images2] completed=1"
    exit 0
  fi

  sleep 3
done

echo "[images2] completed=0"
exit 1
