#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT/splatnav-official"
if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  UPSTREAM_DIR="/data/howard/repos/splatnav-official"
fi

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  echo "Upstream repo missing: $UPSTREAM_DIR"
  exit 1
fi

git -C "$UPSTREAM_DIR" fetch --all --tags
echo "commit=$(git -C "$UPSTREAM_DIR" rev-parse --short HEAD)"
echo "branch=$(git -C "$UPSTREAM_DIR" branch --show-current)"
