#!/bin/bash
#SBATCH --job-name=curriculum_tutor
#SBATCH --partition=qigl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=30-00:00:00
#SBATCH --output=slurm_output/prepare_%j.out
#SBATCH --error=slurm_output/prepare_%j.err
set -euo pipefail
module load anaconda3-2024.10-1
module load cuda-12.4
cd "${SLURM_SUBMIT_DIR:?}"
source hpc/environment.sh
export CUDA_VISIBLE_DEVICES=''
exec "$HPC_SIM_PYTHON" -u hpc/prepare.py --out "$HPC_OUTPUT/preparation" --workers 16
