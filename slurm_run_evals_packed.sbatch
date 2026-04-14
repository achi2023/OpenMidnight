#!/usr/bin/env bash
#SBATCH --job-name=OM_eva_packed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=72:00:00
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
RUN_EVALS_SCRIPT="${RUN_EVALS_SCRIPT:-${SCRIPT_DIR}/run_evals.sh}"
DEFAULT_SWEEP_DIR="${DEFAULT_SWEEP_DIR:-/data/anis/sweeps/vit_base_blr3e-4/eval}"
DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT:-${HOME}/openmidnight_eval_results}"
CKPT_PARALLELISM="${CKPT_PARALLELISM:-2}"
TASK_PARALLELISM="${TASK_PARALLELISM:-1}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-0}"
N_DATA_WORKERS="${N_DATA_WORKERS:-4}"
TASK_DATASETS="${TASK_DATASETS:-bach,bracs,breakhist,crc,gleason,mhist,pcam_10,pcam,consep,monusac}"
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov2_vitb14_reg}"
MODEL_NAME="${MODEL_NAME:-vitb14_reg}"
IN_FEATURES="${IN_FEATURES:-768}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-64}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
TORCH_HOME_BASE="${TORCH_HOME_BASE:-/tmp/${USER}/torch/${SLURM_JOB_ID:-$$}}"
TRITON_HOME_BASE="${TRITON_HOME_BASE:-/tmp/${USER}/triton/${SLURM_JOB_ID:-$$}}"
ALLOW_OVERSUBSCRIBE="${ALLOW_OVERSUBSCRIBE:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<EOF
Usage:
  sbatch ${SCRIPT_SOURCE} [sweep_dir_or_eval_dir] [output_root]

Defaults:
  sweep_dir_or_eval_dir: ${DEFAULT_SWEEP_DIR}
  output_root: ${DEFAULT_OUTPUT_ROOT}

Packed-run env knobs:
  CKPT_PARALLELISM=${CKPT_PARALLELISM}
  TASK_PARALLELISM=${TASK_PARALLELISM}
  MAX_CHECKPOINTS=${MAX_CHECKPOINTS}   # 0 means use all checkpoints
  TASK_DATASETS=${TASK_DATASETS}
  N_DATA_WORKERS=${N_DATA_WORKERS}
  ALLOW_OVERSUBSCRIBE=${ALLOW_OVERSUBSCRIBE}
  BACKBONE_MODEL=${BACKBONE_MODEL}
  MODEL_NAME=${MODEL_NAME}
  IN_FEATURES=${IN_FEATURES}
  PREDICT_BATCH_SIZE=${PREDICT_BATCH_SIZE}
  TORCH_HOME_BASE=${TORCH_HOME_BASE}
  TRITON_HOME_BASE=${TRITON_HOME_BASE}

Notes:
  - This launcher keeps one GPU allocation and runs multiple independent EVA evals on that GPU.
  - The defaults are tuned for vit_base checkpoints under /data/anis/sweeps/vit_base_blr3e-4/eval.
  - Set TASK_DATASETS=ALL to restore the full dataset list from run_evals.sh.
EOF
}


resolve_eval_dir() {
  local sweep_dir="$1"
  if [[ -d "${sweep_dir%/}/eval" ]]; then
    printf '%s\n' "${sweep_dir%/}/eval"
    return
  fi
  if [[ "$(basename "${sweep_dir%/}")" == "eval" ]]; then
    printf '%s\n' "${sweep_dir%/}"
    return
  fi

  echo "Could not find an eval directory under: ${sweep_dir}" >&2
  exit 1
}

collect_checkpoints() {
  local eval_dir="$1"
  local -n checkpoints_ref="$2"
  local total_checkpoints
  local sample_idx
  local -a all_checkpoints=()

  mapfile -t all_checkpoints < <(find "${eval_dir}" -mindepth 2 -maxdepth 2 -type f -name 'teacher_checkpoint.pth' | sort -V)
  if [[ "${#all_checkpoints[@]}" -eq 0 ]]; then
    echo "No teacher checkpoints found under: ${eval_dir}" >&2
    exit 1
  fi

  total_checkpoints="${#all_checkpoints[@]}"
  if (( MAX_CHECKPOINTS <= 0 || total_checkpoints <= MAX_CHECKPOINTS )); then
    checkpoints_ref=("${all_checkpoints[@]}")
    return
  fi

  if (( MAX_CHECKPOINTS == 1 )); then
    checkpoints_ref=("${all_checkpoints[$(( total_checkpoints - 1 ))]}")
    return
  fi

  checkpoints_ref=()
  for ((sample_idx=0; sample_idx<MAX_CHECKPOINTS; sample_idx++)); do
    checkpoints_ref+=("${all_checkpoints[$(( sample_idx * (total_checkpoints - 1) / (MAX_CHECKPOINTS - 1) ))]}")
  done
}

main() {
  if [[ $# -gt 2 ]]; then
    usage
    exit 1
  fi

  local sweep_input="${1:-${DEFAULT_SWEEP_DIR}}"
  local base_output_root="${2:-${DEFAULT_OUTPUT_ROOT}}"
  local eval_dir
  local sweep_name
  local running=0
  local failures=0
  local total_eva_processes=0
  local checkpoint_path
  local checkpoint_tag
  local checkpoint_output_root
  local -a checkpoints=()

  if [[ ! -d "${sweep_input}" ]]; then
    echo "Sweep directory not found: ${sweep_input}" >&2
    exit 1
  fi
  if [[ ! -x "${RUN_EVALS_SCRIPT}" ]]; then
    echo "Run script not found or not executable: ${RUN_EVALS_SCRIPT}" >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Virtualenv not found: ${REPO_ROOT}/.venv/bin/activate" >&2
    exit 1
  fi

  eval_dir="$(resolve_eval_dir "${sweep_input}")"
  collect_checkpoints "${eval_dir}" checkpoints
  sweep_name="$(basename "$(dirname "${eval_dir}")")"
  sweep_name="${sweep_name//[^[:alnum:]_.-]/_}"
  total_eva_processes="$((CKPT_PARALLELISM * TASK_PARALLELISM))"

  if (( total_eva_processes > 4 )) && [[ "${ALLOW_OVERSUBSCRIBE}" != "1" ]]; then
    cat >&2 <<EOF
Refusing to launch ${total_eva_processes} concurrent EVA processes on one GPU.
Current settings:
  CKPT_PARALLELISM=${CKPT_PARALLELISM}
  TASK_PARALLELISM=${TASK_PARALLELISM}

This is the failure mode behind the CUDA OOMs in job 33308.
Use a smaller product, for example:
  CKPT_PARALLELISM=2 TASK_PARALLELISM=2
or override explicitly with:
  ALLOW_OVERSUBSCRIBE=1
EOF
    exit 1
  fi

  if (( TASK_PARALLELISM > 1 && N_DATA_WORKERS > 2 )); then
    echo "Warning: TASK_PARALLELISM=${TASK_PARALLELISM} with N_DATA_WORKERS=${N_DATA_WORKERS} may bottleneck on CPU/shared memory." >&2
    echo "Warning: For parallel dataset evals, prefer N_DATA_WORKERS=1 or 2." >&2
  fi

  mkdir -p "${TORCH_HOME_BASE}/shared" "${TRITON_HOME_BASE}"
  export TORCH_HOME="${TORCH_HOME_BASE}/shared"
  export TRITON_HOME="${TRITON_HOME_BASE}/shared"
  export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
  mkdir -p "${TRITON_CACHE_DIR}"

  echo "Starting packed EVA sweep at $(date -Iseconds)"
  echo "Eval dir: ${eval_dir}"
  echo "Output root: ${base_output_root%/}/${sweep_name}"
  echo "Checkpoints: ${#checkpoints[@]}"
  echo "CKPT_PARALLELISM=${CKPT_PARALLELISM}"
  echo "TASK_PARALLELISM=${TASK_PARALLELISM}"
  echo "TOTAL_EVA_PROCESSES=${total_eva_processes}"
  echo "TASK_DATASETS=${TASK_DATASETS}"
  echo "BACKBONE_MODEL=${BACKBONE_MODEL}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "IN_FEATURES=${IN_FEATURES}"
  echo "PREDICT_BATCH_SIZE=${PREDICT_BATCH_SIZE}"
  echo "N_DATA_WORKERS=${N_DATA_WORKERS}"
  echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "TORCH_HOME=${TORCH_HOME}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

  source "${REPO_ROOT}/.venv/bin/activate"
  echo "Prewarming torch.hub model ${BACKBONE_MODEL}"
  python -c "import torch; torch.hub.load('facebookresearch/dinov2', '${BACKBONE_MODEL}', trust_repo=True)"

  for checkpoint_path in "${checkpoints[@]}"; do
    while (( running >= CKPT_PARALLELISM )); do
      if ! wait -n; then
        failures="$((failures + 1))"
      fi
      running="$((running - 1))"
    done

    checkpoint_tag="$(basename "$(dirname "${checkpoint_path}")")"
    checkpoint_output_root="${base_output_root%/}/${sweep_name}/${checkpoint_tag}"
    mkdir -p "${checkpoint_output_root}"

    echo "Launching ${checkpoint_tag}"
    (
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
      export NUM_DEVICES=1
      export N_DATA_WORKERS
      export TASK_PARALLELISM
      export TASK_DATASETS
      export BACKBONE_MODEL
      export MODEL_NAME
      export IN_FEATURES
      export PREDICT_BATCH_SIZE
      export PYTORCH_CUDA_ALLOC_CONF
      export OMP_NUM_THREADS
      export MKL_NUM_THREADS
      export OPENBLAS_NUM_THREADS
      export TORCH_HOME="${TORCH_HOME_BASE}/shared"
      export TRITON_HOME="${TRITON_HOME_BASE}/${checkpoint_tag}"
      export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
      mkdir -p "${TRITON_CACHE_DIR}"
      exec "${RUN_EVALS_SCRIPT}" "${checkpoint_path}" "${checkpoint_output_root}"
    ) &

    running="$((running + 1))"
  done

  while (( running > 0 )); do
    if ! wait -n; then
      failures="$((failures + 1))"
    fi
    running="$((running - 1))"
  done

  echo "Completed packed EVA sweep at $(date -Iseconds)"
  if (( failures > 0 )); then
    echo "Packed sweep finished with ${failures} failed checkpoint eval(s)." >&2
    exit 1
  fi
}

main "$@"
