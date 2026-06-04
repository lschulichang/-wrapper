#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="/data/howard/miniconda3"
ENV_NAME="${1:-splatnav}"
DEST_ROOT="${2:-/data/howard/splatnav/datasets/splatnav_official}"
DATA_URL="https://drive.google.com/drive/folders/1K0zfpuAti43YIBK5APFd-Yv73CvljgMC?usp=sharing"

mkdir -p "$DEST_ROOT"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Downloading official assets to: $DEST_ROOT"
echo "Source: $DATA_URL"

gdown --folder "$DATA_URL" --continue --remaining-ok -O "$DEST_ROOT"

echo
echo "Download complete. Top-level:"
find "$DEST_ROOT" -maxdepth 2 -type d | head -n 40
