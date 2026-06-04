#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/data/howard/splatnav/datasets/splatnav_official}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/splatnav-official"
if [[ ! -d "$UPSTREAM/.git" ]]; then
  UPSTREAM="/data/howard/repos/splatnav-official"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root does not exist: $DATA_ROOT"
  exit 1
fi

if [[ ! -d "$UPSTREAM/.git" ]]; then
  echo "Upstream repo missing: $UPSTREAM"
  exit 1
fi

if [[ -d "$DATA_ROOT/training/data" ]]; then
  ln -sfn "$DATA_ROOT/training/data" "$UPSTREAM/data"
  echo "linked: $UPSTREAM/data -> $DATA_ROOT/training/data"
fi

if [[ -d "$DATA_ROOT/training/outputs" ]]; then
  ln -sfn "$DATA_ROOT/training/outputs" "$UPSTREAM/outputs"
  echo "linked: $UPSTREAM/outputs -> $DATA_ROOT/training/outputs"
fi

if [[ -d "$DATA_ROOT/traj" ]]; then
  ln -sfn "$DATA_ROOT/traj" "$UPSTREAM/official_traj"
  echo "linked: $UPSTREAM/official_traj -> $DATA_ROOT/traj"
fi

echo
echo "Discovered config files:"
find "$UPSTREAM/outputs" -type f -name "config.yml" 2>/dev/null | head -n 50 || true
