#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gdown_log_file> <scene_root>"
  echo "Example: $0 runs/2026-03-29_xxx/stdout.log /data/howard/splatnav/datasets/splatnav_min/splatnav11202024"
  exit 1
fi

LOG_FILE="$1"
SCENE_ROOT="$2"
MAX_RETRIES="${MAX_RETRIES:-8}"

if [[ ! -f "$LOG_FILE" ]]; then
  echo "log file not found: $LOG_FILE"
  exit 1
fi

mkdir -p "$SCENE_ROOT/images"

TMP_MAP="$(mktemp)"
trap 'rm -f "$TMP_MAP"' EXIT

python3 - "$LOG_FILE" "$TMP_MAP" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

pattern = re.compile(r"Processing file ([A-Za-z0-9_-]+) ([^\s]+)")
entries = {}
for line in log_path.read_text(errors="ignore").splitlines():
    m = pattern.search(line)
    if not m:
        continue
    file_id, name = m.group(1), m.group(2)
    if name.startswith("frame_") and name.endswith(".png"):
        entries.setdefault(name, file_id)
    elif name in ("transforms.json", "sparse_pc.ply"):
        entries.setdefault(name, file_id)

with out_path.open("w", encoding="utf-8") as f:
    for name in sorted(entries.keys()):
        f.write(f"{name}\t{entries[name]}\n")
PY

download_with_retry() {
  local file_id="$1"
  local target="$2"
  local ok=0

  for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    echo "[fill] target=$target attempt=$attempt"
    if gdown "https://drive.google.com/uc?id=${file_id}" --continue -O "$target"; then
      ok=1
      break
    fi
    sleep 2
  done

  if [[ "$ok" != "1" ]]; then
    echo "[fill] failed target=$target"
    return 1
  fi
}

while IFS=$'\t' read -r name file_id; do
  [[ -z "$name" ]] && continue

  target=""
  if [[ "$name" == frame_*.png ]]; then
    target="$SCENE_ROOT/images/$name"
  elif [[ "$name" == "transforms.json" || "$name" == "sparse_pc.ply" ]]; then
    target="$SCENE_ROOT/$name"
  else
    continue
  fi

  if [[ -f "$target" ]]; then
    continue
  fi

  download_with_retry "$file_id" "$target"
done < "$TMP_MAP"

frame_count="$(find "$SCENE_ROOT/images" -type f -name 'frame_*.png' | wc -l | tr -d ' ')"
has_transforms=0
has_sparse_pc=0
[[ -f "$SCENE_ROOT/transforms.json" ]] && has_transforms=1
[[ -f "$SCENE_ROOT/sparse_pc.ply" ]] && has_sparse_pc=1

echo "[fill] frame_count=${frame_count}"
echo "[fill] has_transforms=${has_transforms}"
echo "[fill] has_sparse_pc=${has_sparse_pc}"

if [[ "$frame_count" -ge 50 && "$has_transforms" == "1" && "$has_sparse_pc" == "1" ]]; then
  echo "[fill] completed=1"
  exit 0
fi

echo "[fill] completed=0"
exit 1
