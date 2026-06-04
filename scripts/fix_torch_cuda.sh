#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="/data/howard/miniconda3"
ENV_NAME="${1:-splatnav}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

# Pin to a CUDA runtime compatible with current driver (without changing system CUDA/driver).
python -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1

env -u LD_LIBRARY_PATH python - << 'PY'
import torch
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY
