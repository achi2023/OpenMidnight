#!/usr/bin/env bash
#SBATCH --job-name=dinov2-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --time=infinite
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err
#SBATCH --partition=main
#SBATCH --account=training

set -euo pipefail

# Required sweep params (pass via sbatch --export)
: "${MODEL_ARCH:?Set MODEL_ARCH (vit_small|vit_base|vit_large|vit_giant2)}"
: "${BASE_LR:?Set BASE_LR (for example 1e-4)}"

# Optional overrides
CONFIG_FILE="${CONFIG_FILE:-./dinov2/configs/train/vitg14_reg4.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./sweeps}"
JOB_TAG="${JOB_TAG:-${MODEL_ARCH}_blr${BASE_LR}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${JOB_TAG}"
RESUME="${RESUME:-False}"

case "${MODEL_ARCH}" in
  vit_small|vit_base|vit_large) DEFAULT_FFN_LAYER="mlp" ;;
  vit_giant2) DEFAULT_FFN_LAYER="swiglufused" ;;
  *)
    echo "Unsupported MODEL_ARCH=${MODEL_ARCH}"
    exit 1
    ;;
esac

case "${MODEL_ARCH}" in
  vit_small) DEFAULT_DROP_PATH_RATE="0.1" ;;
  vit_base) DEFAULT_DROP_PATH_RATE="0.2" ;;
  vit_large) DEFAULT_DROP_PATH_RATE="0.3" ;;
  vit_giant2) DEFAULT_DROP_PATH_RATE="0.4" ;;
esac

FFN_LAYER="${FFN_LAYER:-${DEFAULT_FFN_LAYER}}"
DROP_PATH_RATE="${DROP_PATH_RATE:-${DEFAULT_DROP_PATH_RATE}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${RESUME}" == "True" ]]; then
  RESUME_FLAG=""
else
  RESUME_FLAG="--no-resume"
fi
mkdir -p "${OUTPUT_DIR}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL_ARCH=${MODEL_ARCH}"
echo "BASE_LR=${BASE_LR}"
echo "FFN_LAYER=${FFN_LAYER}"
echo "DROP_PATH_RATE=${DROP_PATH_RATE}"
echo "RESUME=${RESUME}"


export TORCH_HOME="/tmp/${USER}/torch/${SLURM_JOB_ID}"
export TRITON_HOME="/tmp/${USER}/triton/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
mkdir -p "${TORCH_HOME}" "${TRITON_CACHE_DIR}"


uv run torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --node_rank 0 \
  dinov2/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RESUME_FLAG} \
  student.arch="${MODEL_ARCH}" \
  student.ffn_layer="${FFN_LAYER}" \
  student.drop_path_rate="${DROP_PATH_RATE}" \
  optim.base_lr="${BASE_LR}"
