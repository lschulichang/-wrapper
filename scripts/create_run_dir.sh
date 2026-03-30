#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <topic>"
  echo "Example: $0 env_check"
  exit 1
fi

TOPIC="$1"
ROOT="/home/howard/Splat-Nav"
RUNS_DIR="$ROOT/runs"
TIMESTAMP="$(date +%Y-%m-%d_%H%M)"
RUN_DIR="$RUNS_DIR/${TIMESTAMP}_${TOPIC}"
UPSTREAM_DIR="$ROOT/splatnav-official"
if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  UPSTREAM_DIR="/data/howard/repos/splatnav-official"
fi

mkdir -p "$RUN_DIR/assets"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "user=$(whoami)"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  echo "python=$(python3 --version 2>/dev/null || true)"
  if [[ -d "$UPSTREAM_DIR/.git" ]]; then
    echo "splatnav_official_commit=$(git -C "$UPSTREAM_DIR" rev-parse --short HEAD)"
  fi
} > "$RUN_DIR/env.txt"

cat > "$RUN_DIR/notes.md" << 'EOF'
# Notes

- Goal:
- Result:
- Issues:
- Next:
EOF

touch "$RUN_DIR/cmd.txt" "$RUN_DIR/stdout.log"

echo "$RUN_DIR"
