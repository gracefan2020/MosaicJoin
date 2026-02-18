#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem 128GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --account=torch_pr_66_tandon_advanced
#SBATCH --comment="preemption=yes;requeue=true"

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(pwd)"
fi

ENV_PREFIX="${SCRIPT_DIR}/penv"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK="${1:-all}"
OVERRIDE_CHECKPOINT="${2:-${SNOOPY_CHECKPOINT_PATH:-}}"
TOP_K="${SNOOPY_TOP_K:-50}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${SCRIPT_DIR}/checkpoints}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/model}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-${SCRIPT_DIR}/precomputed/WDC}"

CKPT_OPENDATA="${CKPT_OPENDATA:-${CHECKPOINTS_DIR}/opendata/t=0.2_gendata_mat_leveltrained.pth}"
CKPT_WIKITABLE="${CKPT_WIKITABLE:-${CHECKPOINTS_DIR}/WikiTable/t=0.2_gendata_mat_leveltrained.pth}"
CKPT_WDC="${CKPT_WDC:-${CHECKPOINTS_DIR}/WDC/t=0.2_gendata_mat_leveltrained.pth}"

FASTTEXT_MODEL_PATH="${FASTTEXT_MODEL_PATH:-${MODEL_DIR}/cc.en.300.bin}"
PRECOMPUTED_TARGET_NPY="${PRECOMPUTED_TARGET_NPY:-${PRECOMPUTED_DIR}/target.npy}"
PRECOMPUTED_QUERY_NPY="${PRECOMPUTED_QUERY_NPY:-${PRECOMPUTED_DIR}/query.npy}"

module purge
module load anaconda3/2025.06

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

echo "=== GPU CHECK ==="
nvidia-smi || true

cd "${SCRIPT_DIR}"
mkdir -p "${SCRIPT_DIR}/cache"

run_benchmark () {
  local bench="$1"
  local dataset_dir=""
  local query_source=""
  local checkpoint_path=""
  local out_csv="${SCRIPT_DIR}/snoopy_ft_${bench}.csv"
  local index_cache="${SCRIPT_DIR}/cache/snoopy-${bench}.pkl"

  case "${bench}" in
    autofj)
      dataset_dir="${REPO_ROOT}/datasets/autofj"
      query_source="${dataset_dir}/query_columns.csv"
      checkpoint_path="${CKPT_OPENDATA}"
      ;;
    autofj-wdc)
      dataset_dir="${REPO_ROOT}/datasets/autofj-wdc"
      query_source="${dataset_dir}/query_columns.csv"
      checkpoint_path="${CKPT_WDC}"
      ;;
    freyja)
      dataset_dir="${REPO_ROOT}/datasets/freyja"
      query_source="${dataset_dir}/query_columns.csv"
      checkpoint_path="${CKPT_WDC}"
      ;;
    freyja-wdc)
      dataset_dir="${REPO_ROOT}/datasets/freyja-wdc"
      query_source="${dataset_dir}/query_columns.csv"
      checkpoint_path="${CKPT_WDC}"
      ;;
    wt)
      dataset_dir="${REPO_ROOT}/datasets/wt"
      query_source="${dataset_dir}/query_columns.csv"
      checkpoint_path="${CKPT_WIKITABLE}"
      ;;
    wt-wdc)
      dataset_dir="${REPO_ROOT}/datasets/wt-wdc"
      query_source="${dataset_dir}/query_columns.csv"
      checkpoint_path="${CKPT_WDC}"
      ;;
    *)
      echo "Unsupported benchmark: ${bench}"
      echo "Supported: all, autofj, autofj-wdc, freyja, freyja-wdc, wt, wt-wdc"
      return 1
      ;;
  esac

  if [[ -n "${OVERRIDE_CHECKPOINT}" ]]; then
    checkpoint_path="${OVERRIDE_CHECKPOINT}"
  fi

  echo "=============================================="
  echo "Benchmark: ${bench}"
  echo "Dataset: ${dataset_dir}"
  echo "Query source: ${query_source}"
  echo "Checkpoint: ${checkpoint_path}"
  echo "fastText model: ${FASTTEXT_MODEL_PATH}"
  echo "Precomputed target npy: ${PRECOMPUTED_TARGET_NPY}"
  echo "Precomputed query npy: ${PRECOMPUTED_QUERY_NPY}"
  echo "Output: ${out_csv}"
  echo "Index cache: ${index_cache}"

  python run_snoopy_baseline.py \
    --dataset_dir "${dataset_dir}" \
    --query_source "${query_source}" \
    --out_csv "${out_csv}" \
    --checkpoint_path "${checkpoint_path}" \
    --index_cache "${index_cache}" \
    --device auto \
    --top_k "${TOP_K}" \
    --fasttext_model_path "${FASTTEXT_MODEL_PATH}" \
    --precomputed_target_npy "${PRECOMPUTED_TARGET_NPY}" \
    --precomputed_query_npy "${PRECOMPUTED_QUERY_NPY}" \
    ${SNOOPY_EXTRA_ARGS:-}
}

if [[ "${BENCHMARK}" == "all" ]]; then
  for bench in autofj autofj-wdc freyja freyja-wdc wt wt-wdc; do
    run_benchmark "${bench}"
  done
else
  run_benchmark "${BENCHMARK}"
fi
