#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load anaconda3-2024.10-1
export CONDA_PKGS_DIRS="$PWD/cache/conda" PIP_CACHE_DIR="$PWD/cache/pip" TMPDIR="$PWD/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
if [[ ! -x "$PWD/.venv-training/bin/python" ]]; then conda create -y -p "$PWD/.venv-training" python=3.10 pip; fi
PYTHON="$PWD/.venv-training/bin/python"
"$PYTHON" -m pip install -r hpc/requirements-training.txt
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  "$PYTHON" -m pip install torch==2.8.0 --index-url "$TORCH_INDEX_URL"
else
  "$PYTHON" -m pip install torch==2.8.0
fi
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > cache/training_installed_versions.txt
printf 'Training environment ready: %s\nSimulation uses the existing SMAAC Python3.13 environment.\n' "$PYTHON"
