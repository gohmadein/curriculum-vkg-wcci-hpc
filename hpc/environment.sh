#!/bin/bash
# Source only after entering the repository root. Slurm retains GPU assignment.
export CURRICULUM_ROOT="$PWD"
export HPC_SIM_PYTHON="${HPC_SIM_PYTHON:-$(dirname "$PWD")/smaac-vkg-wcci-hpc/.venv-conda/bin/python}"
export HPC_TRAIN_PYTHON="${HPC_TRAIN_PYTHON:-$PWD/.venv-training/bin/python}"
export VKG_DATA_ROOT="${VKG_DATA_ROOT:-$(dirname "$PWD")/smaac-vkg-wcci-hpc/l2rpn_wcci_2022-frozen.tar_2026101318291010}"
export HPC_OUTPUT="${HPC_OUTPUT:-$PWD/runs/curriculum_wcci_v1}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/runtime:$PWD/upstream:$PWD/training:$PWD/hpc"
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache" MPLCONFIGDIR="$PWD/cache/matplotlib" NUMBA_CACHE_DIR="$PWD/cache/numba" TORCH_HOME="$PWD/cache/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCH_HOME" "$HPC_OUTPUT"
if [[ ! -x "$HPC_SIM_PYTHON" ]]; then echo "Missing simulation Python: $HPC_SIM_PYTHON" >&2; return 2; fi
if [[ ! -d "$VKG_DATA_ROOT/l2rpn_wcci_2022/chronics" ]]; then echo "Dataset parent incorrect: $VKG_DATA_ROOT" >&2; return 2; fi
