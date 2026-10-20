#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cat runs/submission.json
squeue -u "$USER"
# Print persisted status; no heavyweight Python import on the login node.
find "${HPC_OUTPUT:-$PWD/runs/curriculum_wcci_v1}" -name status.json -o -name 'status_s*.json' -o -name run_meta.json
