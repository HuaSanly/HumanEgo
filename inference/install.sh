#!/usr/bin/env bash
# Usage: bash inference/install.sh /absolute/path/to/new/venv /path/to/python3.12
set -euo pipefail
inference_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
inference_env="${1:?Provide the path of a NEW inference environment}"
inference_python="${2:-python3.12}"
if [[ -e "$inference_env" ]]; then
    echo "Refusing to alter an existing environment: $inference_env" >&2
    exit 1
fi
"$inference_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
"$inference_python" -m venv "$inference_env"
"$inference_env/bin/python" -m pip install 'pip==26.2.2' 'setuptools==81.0.0'
"$inference_env/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
SAM2_BUILD_CUDA=0 "$inference_env/bin/python" -m pip install --no-build-isolation -r "$inference_root/requirements.in" -c "$inference_root/constraints.txt"
"$inference_env/bin/python" -m pip install --no-deps "$inference_root"
"$inference_env/bin/python" -m pip check
"$inference_env/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
