#!/usr/bin/env bash
#SBATCH --job-name=OM_eva
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=/admin/home/%u/OpenMidnight/slurms_eval/%x_%j.out
#SBATCH --error=/admin/home/%u/OpenMidnight/slurms_eval/%x_%j.err

set -euo pipefail

SCRIPT_SOURCE="${SLURM_SCRIPT_SOURCE:-${BASH_SOURCE[0]}}"
if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_SCRIPT_SOURCE:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  SLURM_COMMAND_PATH="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | sed -n 's/.* Command=\([^ ]*\).*/\1/p' | head -n 1)"
  if [[ -n "${SLURM_COMMAND_PATH}" && -f "${SLURM_COMMAND_PATH}" ]]; then
    SCRIPT_SOURCE="${SLURM_COMMAND_PATH}"
  fi
fi

SCRIPT_DIR="${EVA_PROBE_DIR:-$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd -P)}"
REPO_ROOT="${OPENMIDNIGHT_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
THIS_SCRIPT="${SCRIPT_DIR}/$(basename "${SCRIPT_SOURCE}")"
RUN_EVALS_SCRIPT="${RUN_EVALS_SCRIPT:-${SCRIPT_DIR}/run_evals.sh}"
DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT:-${HOME}/openmidnight_eval_results}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-12}"

usage() {
  cat <<EOF
Usage:
  Launch one eval job per teacher checkpoint in a sweep:
    ${THIS_SCRIPT} /absolute/path/to/sweep_dir [output_root]

  Submit a single checkpoint directly:
    ${SBATCH_BIN} --export=ALL,CKPT_PATH=/absolute/path/to/teacher_checkpoint.pth,OUTPUT_ROOT=/absolute/path/to/output_root ${THIS_SCRIPT}

Notes:
  - Direct execution is the preferred sweep-launch mode because it does not hold a launcher allocation.
  - 'sbatch ${THIS_SCRIPT} /path/to/sweep_dir' also works, but it launches child jobs from inside the batch job.
  - When more than ${MAX_CHECKPOINTS} checkpoints are present, the launcher samples ${MAX_CHECKPOINTS}
    checkpoints evenly from latest to earliest, inclusive.
EOF
}

submit_sweep() {
  local sweep_dir="$1"
  local base_output_root="${2:-${DEFAULT_OUTPUT_ROOT}}"
  local eval_dir
  local sweep_name
  local checkpoint_dir
  local checkpoint_tag
  local checkpoint_path
  local checkpoint_output_root
  local export_args
  local job_id
  local total_checkpoints
  local submitted_count=0
  local target_checkpoints
  local sample_idx
  local -a all_checkpoints=()
  local -a checkpoints=()

  if [[ ! -d "${sweep_dir}" ]]; then
    echo "Sweep directory not found: ${sweep_dir}"
    exit 1
  fi

  if [[ -d "${sweep_dir%/}/eval" ]]; then
    eval_dir="${sweep_dir%/}/eval"
  elif [[ "$(basename "${sweep_dir%/}")" == "eval" ]]; then
    eval_dir="${sweep_dir%/}"
    sweep_dir="$(dirname "${eval_dir}")"
  else
    echo "Could not find an eval directory under: ${sweep_dir}"
    exit 1
  fi

  if ! command -v "${SBATCH_BIN}" >/dev/null 2>&1; then
    echo "sbatch executable not found: ${SBATCH_BIN}"
    exit 1
  fi

  mapfile -t all_checkpoints < <(find "${eval_dir}" -mindepth 2 -maxdepth 2 -type f -name 'teacher_checkpoint.pth' | sort -rV)
  if [[ "${#all_checkpoints[@]}" -eq 0 ]]; then
    echo "No teacher checkpoints found under: ${eval_dir}"
    exit 1
  fi

  total_checkpoints="${#all_checkpoints[@]}"
  target_checkpoints="${MAX_CHECKPOINTS}"
  if (( target_checkpoints < 1 )); then
    echo "MAX_CHECKPOINTS must be >= 1, got: ${MAX_CHECKPOINTS}"
    exit 1
  fi

  if (( total_checkpoints <= target_checkpoints )); then
    checkpoints=("${all_checkpoints[@]}")
  elif (( target_checkpoints == 1 )); then
    checkpoints=("${all_checkpoints[0]}")
  else
    for ((sample_idx=0; sample_idx<target_checkpoints; sample_idx++)); do
      checkpoints+=("${all_checkpoints[$(( sample_idx * (total_checkpoints - 1) / (target_checkpoints - 1) ))]}")
    done
  fi

  sweep_name="$(basename "${sweep_dir%/}")"
  sweep_name="${sweep_name//[^[:alnum:]_.-]/_}"

  echo "Found ${total_checkpoints} checkpoints under ${eval_dir}"
  echo "Submitting ${#checkpoints[@]} eval jobs from latest to earliest"
  echo "Results root: ${base_output_root%/}/${sweep_name}"

  for checkpoint_path in "${checkpoints[@]}"; do
    checkpoint_dir="$(basename "$(dirname "${checkpoint_path}")")"
    checkpoint_tag="${checkpoint_dir//[^[:alnum:]_.-]/_}"
    checkpoint_output_root="${base_output_root%/}/${sweep_name}/${checkpoint_tag}"
    export_args="ALL,CKPT_PATH=${checkpoint_path},OUTPUT_ROOT=${checkpoint_output_root},JOB_TAG=${checkpoint_tag},RUN_EVALS_SCRIPT=${RUN_EVALS_SCRIPT},EVA_PROBE_DIR=${SCRIPT_DIR},OPENMIDNIGHT_REPO_ROOT=${REPO_ROOT},SLURM_SCRIPT_SOURCE=${THIS_SCRIPT}"

    echo "Submitting ${checkpoint_tag}"
    job_id="$("${SBATCH_BIN}" --parsable --job-name="OM_eva_${sweep_name}_${checkpoint_tag}" --export="${export_args}" "${THIS_SCRIPT}")"
    job_id="${job_id%%;*}"
    echo "  job_id=${job_id} checkpoint=${checkpoint_path}"
    submitted_count="$((submitted_count + 1))"
  done

  echo "Submitted ${submitted_count} eval jobs."
}

run_single_eval() {
  : "${CKPT_PATH:?Set CKPT_PATH to an absolute teacher checkpoint path}"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}"
    exit 1
  fi

  local output_root="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
  mkdir -p "${output_root}"

  if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Virtualenv not found: ${REPO_ROOT}/.venv/bin/activate"
    exit 1
  fi

  source "${REPO_ROOT}/.venv/bin/activate"
  cd "${SCRIPT_DIR}"

  echo "Starting eval job"
  echo "CKPT_PATH=${CKPT_PATH}"
  echo "OUTPUT_ROOT=${output_root}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  fi

  "${RUN_EVALS_SCRIPT}" "${CKPT_PATH}" "${output_root}"
}

main() {
  if [[ $# -gt 0 ]]; then
    if [[ $# -gt 2 ]]; then
      usage
      exit 1
    fi

    submit_sweep "$1" "${2:-${DEFAULT_OUTPUT_ROOT}}"
    return
  fi

  if [[ -n "${CKPT_PATH:-}" ]]; then
    run_single_eval
    return
  fi

  usage
  exit 1
}

main "$@"
