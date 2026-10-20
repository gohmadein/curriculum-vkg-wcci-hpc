#!/bin/bash
# Submit the entire dependency chain once: shared Tutor -> two seniors -> summary/export.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p slurm_output runs tmp cache
source hpc/environment.sh
if [[ ! -x "$HPC_TRAIN_PYTHON" ]]; then echo "Run bash hpc/setup_training.sh before submission" >&2; exit 2; fi
if [[ -f runs/submission.json ]]; then echo 'Existing submission record; check squeue and runs/submission.json before resubmitting.' >&2; exit 2; fi
prep_reply=$(sbatch --parsable hpc/prepare.sbatch)
prep_id=${prep_reply%%;*}
if [[ ! "$prep_id" =~ ^[0-9]+$ ]]; then echo "Cannot parse preparation job id: $prep_reply" >&2; exit 2; fi
printf '{"preparation_job":"%s","state":"preparation_submitted"}\n' "$prep_id" > runs/submission.json
train_reply=$(sbatch --parsable --dependency="afterok:$prep_id" hpc/formal.sbatch)
train_id=${train_reply%%;*}
if [[ ! "$train_id" =~ ^[0-9]+$ ]]; then echo "Cannot parse training job id: $train_reply" >&2; exit 2; fi
printf '{"preparation_job":"%s","training_array":"%s","state":"training_submitted"}\n' "$prep_id" "$train_id" > runs/submission.json
summary_reply=$(sbatch --parsable --dependency="afterok:$train_id" hpc/aggregate.sbatch)
summary_id=${summary_reply%%;*}
if [[ ! "$summary_id" =~ ^[0-9]+$ ]]; then echo "Cannot parse summary job id: $summary_reply" >&2; exit 2; fi
printf '{"preparation_job":"%s","training_array":"%s","summary_job":"%s","state":"submitted"}\n' "$prep_id" "$train_id" "$summary_id" > runs/submission.json
cat runs/submission.json
