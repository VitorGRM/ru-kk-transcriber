#!/usr/bin/env bash
# One-time setup. Creates a virtual environment and installs everything.
#
#   ./setup.sh          CPU-only install
#   ./setup.sh --gpu    also install the CUDA runtime libraries (NVIDIA GPUs)
set -euo pipefail
cd "$(dirname "$0")"

GPU=0
[[ "${1:-}" == "--gpu" ]] && GPU=1

echo "==> Creating the virtual environment"
if command -v uv >/dev/null 2>&1; then
  uv venv .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  python3 -m venv .venv
  PIP=(.venv/bin/python -m pip install --upgrade)
  "${PIP[@]}" pip >/dev/null
fi

echo "==> Installing dependencies"
"${PIP[@]}" -r requirements.txt

if [[ $GPU -eq 1 ]]; then
  echo "==> Installing the CUDA runtime libraries for CTranslate2"
  # CTranslate2 needs cuBLAS and cuDNN 9. Installing them from PyPI avoids a
  # system-wide CUDA toolkit; run.sh puts them on the library path.
  "${PIP[@]}" nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
fi

echo
echo "Setup complete. Start the app with:  ./run.sh"
