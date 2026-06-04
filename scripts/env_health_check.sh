#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="/data/howard/miniconda3"
ENV_NAME="${1:-splatnav}"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "Conda profile script missing: $CONDA_ROOT/etc/profile.d/conda.sh"
  exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "== System =="
uname -a
echo

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo

echo "== Python =="
python --version
echo

echo "== Torch CUDA =="
env -u LD_LIBRARY_PATH python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device_name:", torch.cuda.get_device_name(0))
PY
echo

echo "== Key imports =="
env -u LD_LIBRARY_PATH python - << 'PY'
import importlib
mods = ["nerfstudio", "open3d", "cv2", "clarabel", "cvxpy", "dijkstra3d", "polytope", "unfoldNd", "viser", "lightglue"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL -> {e}")
PY
