#!/bin/bash
#SBATCH --job-name=curriculum_formal
#SBATCH --partition=qigl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00
#SBATCH --array=0-1%2
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm_output/%A_%a.out
#SBATCH --error=slurm_output/%A_%a.err
set -euo pipefail
module load anaconda3-2024.10-1
module load cuda-12.4
cd "${SLURM_SUBMIT_DIR:?}"
source hpc/environment.sh
if [[ ! -x "$HPC_TRAIN_PYTHON" ]]; then echo "Run bash hpc/setup_training.sh first" >&2; exit 2; fi
child_pid=''
stop_child() { if [[ -n "$child_pid" ]]; then kill -USR1 "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi; exit 1; }
trap stop_child USR1 TERM INT
"$HPC_SIM_PYTHON" -u hpc/task.py --seed "${SLURM_ARRAY_TASK_ID:?}" --out "$HPC_OUTPUT" --training-python "$HPC_TRAIN_PYTHON" --eval-workers 18 &
child_pid=$!
wait "$child_pid"
