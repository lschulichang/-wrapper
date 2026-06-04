#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <folder_url> <output_dir> [max_attempts] [min_frame_images]"
  echo "Example: $0 'https://drive.google.com/drive/folders/XXX' /data/howard/splatnav/datasets/scene 20 50"
  exit 1
fi

FOLDER_URL="$1"
OUTPUT_DIR="$2"
MAX_ATTEMPTS="${3:-20}"
MIN_FRAME_IMAGES="${4:-50}"

mkdir -p "$OUTPUT_DIR"

for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
  echo "[retry] attempt=${attempt}"

  if gdown --folder "$FOLDER_URL" --continue --remaining-ok -O "$OUTPUT_DIR"; then
    echo "[retry] gdown_status=ok"
  else
    echo "[retry] gdown_status=fail"
  fi

  frame_count="$(find "$OUTPUT_DIR" -type f -name 'frame_*.png' | wc -l | tr -d ' ')"
  has_transforms=0
  has_sparse_pc=0
  [[ -f "$OUTPUT_DIR/transforms.json" ]] && has_transforms=1
  [[ -f "$OUTPUT_DIR/sparse_pc.ply" ]] && has_sparse_pc=1

  echo "[retry] frame_count=${frame_count}"
  echo "[retry] has_transforms=${has_transforms}"
  echo "[retry] has_sparse_pc=${has_sparse_pc}"

  if [[ "$has_transforms" == "1" && "$has_sparse_pc" == "1" && "$frame_count" -ge "$MIN_FRAME_IMAGES" ]]; then
    echo "[retry] completed=1"
    exit 0
  fi

  sleep 5
done

echo "[retry] completed=0"
exit 1
