#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="/data/howard/miniconda3"
CONDA_BIN="$CONDA_ROOT/bin/conda"
ENV_NAME="${1:-splatnav}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda not found at $CONDA_BIN"
  exit 1
fi

export TMPDIR=/data/howard/splatnav/tmp
export PIP_CACHE_DIR=/data/howard/splatnav/cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y --override-channels -c conda-forge -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

# Nerfstudio is the upstream dependency mentioned in the official Splat-Nav README.
python -m pip install nerfstudio

# Core dependencies referenced by Splat-Nav code.
python -m pip install \
  clarabel \
  dijkstra3d \
  polytope \
  cvxopt \
  cvxpy \
  unfoldNd \
  viser \
  open3d \
  trimesh \
  scipy \
  matplotlib \
  opencv-python \
  imageio \
  sympy \
  tqdm

# LightGlue is used by pose_estimator/utils.py. Install from official repo.
python -m pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"

env -u LD_LIBRARY_PATH python - << 'PY'
import importlib

modules = [
    "torch",
    "nerfstudio",
    "open3d",
    "cv2",
    "clarabel",
    "cvxpy",
    "dijkstra3d",
    "polytope",
    "unfoldNd",
    "viser",
    "lightglue",
]

missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

if missing:
    raise SystemExit(f"Missing imports: {missing}")

print("All key imports passed.")
PY

echo "Environment bootstrap completed: $ENV_NAME"
