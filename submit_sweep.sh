#!/usr/bin/env bash
#SBATCH --job-name=OM_lr_submit_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=infinite
#SBATCH --output=slurms/%j.submit.out
#SBATCH --error=slurms/%j.submit.err
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --requeue
set -euo pipefail

# Sequential submission without holding resources for days:
# each job is submitted with dependency on the prior job.
# DEPENDENCY_TYPE can be "afterany" (default) or "afterok".
source .venv/bin/activate
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterany}"

ARCH_LIST="${ARCH_LIST:-vit_small,vit_base,vit_large,vit_giant2}"
LR_LIST="${LR_LIST:-1e-4,2e-4,4e-4}"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-./run_sweep_1node.sbatch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./sweeps}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Missing sbatch script: ${SBATCH_SCRIPT}"
  exit 1
fi

IFS=',' read -r -a ARCHES <<< "${ARCH_LIST}"
IFS=',' read -r -a LRS <<< "${LR_LIST}"
prev_job_id=""

for arch in "${ARCHES[@]}"; do
  for lr in "${LRS[@]}"; do
    tag="${arch}_blr${lr}"
    export_args="ALL,MODEL_ARCH=${arch},BASE_LR=${lr},OUTPUT_ROOT=${OUTPUT_ROOT},JOB_TAG=${tag}"

    if [[ -z "${prev_job_id}" ]]; then
      echo "Submitting ${tag}"
      job_id="$(sbatch --parsable --export="${export_args}" "${SBATCH_SCRIPT}")"
    else
      echo "Submitting ${tag} (depends on ${prev_job_id})"
      job_id="$(sbatch --parsable --dependency="${DEPENDENCY_TYPE}:${prev_job_id}" --export="${export_args}" "${SBATCH_SCRIPT}")"
    fi

    prev_job_id="${job_id%%;*}"
    echo "  job_id=${prev_job_id}"
  done
done
