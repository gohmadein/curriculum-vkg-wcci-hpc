#!/bin/bash
#SBATCH --job-name=curriculum_summary
#SBATCH --partition=qigl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/summary_%j.out
#SBATCH --error=slurm_output/summary_%j.err
set -euo pipefail
module load anaconda3-2024.10-1
cd "${SLURM_SUBMIT_DIR:?}"
source hpc/environment.sh
export CUDA_VISIBLE_DEVICES=''
"$HPC_SIM_PYTHON" hpc/aggregate.py --out "$HPC_OUTPUT"
"$HPC_SIM_PYTHON" hpc/export_results.py --out "$HPC_OUTPUT" --archive "$PWD/curriculum_wcci_results_${SLURM_JOB_ID}.tar.gz" --include-models
