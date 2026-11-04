#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
assert torch.cuda.is_available(), 'Enable GPU in Kaggle notebook settings first.'
assert CUDA_HOME is not None, 'CUDA toolkit/nvcc is required to build selective scan.'
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name())
PY
# Keep the Kaggle torch/torchvision/CUDA pair. Do not install upstream requirements.txt.
python -m pip install 'setuptools>=70' wheel ninja packaging pytest 'timm==1.0.14' 'einops==0.8.1'
python -m pip install -e .
MAX_JOBS=2 python -m pip install --no-build-isolation ./selective_scan
python -c "import selective_scan_cuda_core; import ultralytics; print(ultralytics.__file__)"
python -m pytest -q tests
python scripts/smoke_test.py --device cuda
