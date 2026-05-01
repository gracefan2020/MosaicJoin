#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 128GB
#SBATCH --gres=gpu:l40s:1
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

SILKMOTH_METRIC="${SILKMOTH_METRIC:-containment}"
SILKMOTH_THRESHOLD="${SILKMOTH_THRESHOLD:-0.5}"
SILKMOTH_TOP_K="${SILKMOTH_TOP_K:-0}"
SILKMOTH_COLUMN_NAME="${SILKMOTH_COLUMN_NAME:-title}"
SILKMOTH_DEFAULT_QUERY_COLUMN="${SILKMOTH_DEFAULT_QUERY_COLUMN:-title}"
SILKMOTH_MAX_DATALAKE_TABLES="${SILKMOTH_MAX_DATALAKE_TABLES:-0}"
SILKMOTH_MAX_QUERIES="${SILKMOTH_MAX_QUERIES:-0}"
SILKMOTH_LOG_LEVEL="${SILKMOTH_LOG_LEVEL:-INFO}"
CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"
SCORE_WORKERS="${SILKMOTH_SCORE_WORKERS:-${CPU_THREADS}}"

module purge
module load anaconda3/2025.06

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"

cd "${SCRIPT_DIR}"

run_benchmark () {
  local bench="$1"
  local dataset_dir=""
  local query_source=""
  local out_csv="${SCRIPT_DIR}/silkmoth_${SILKMOTH_METRIC}_${bench}.csv"

  case "${bench}" in
    autofj)
      dataset_dir="${REPO_ROOT}/datasets/autofj"
      query_source="${dataset_dir}/query_columns.csv"
      ;;
    autofj-wdc)
      dataset_dir="${REPO_ROOT}/datasets/autofj-wdc"
      query_source="${dataset_dir}/query_columns.csv"
      ;;
    freyja)
      dataset_dir="${REPO_ROOT}/datasets/freyja"
      query_source="${dataset_dir}/query_columns.csv"
      ;;
    freyja-wdc)
      dataset_dir="${REPO_ROOT}/datasets/freyja-wdc"
      query_source="${dataset_dir}/query_columns.csv"
      ;;
    wt)
      dataset_dir="${REPO_ROOT}/datasets/wt"
      query_source="${dataset_dir}/query_columns.csv"
      ;;
    wt-wdc)
      dataset_dir="${REPO_ROOT}/datasets/wt-wdc"
      query_source="${dataset_dir}/query_columns.csv"
      ;;
    *)
      echo "Unsupported benchmark: ${bench}"
      echo "Supported: all, autofj, autofj-wdc, freyja, freyja-wdc, wt, wt-wdc"
      return 1
      ;;
  esac

  echo "=============================================="
  echo "Benchmark: ${bench}"
  echo "Dataset: ${dataset_dir}"
  echo "Query source: ${query_source}"
  echo "Metric: ${SILKMOTH_METRIC}"
  echo "Threshold: ${SILKMOTH_THRESHOLD}"
  echo "Column: ${SILKMOTH_COLUMN_NAME}"
  echo "Top-k: ${SILKMOTH_TOP_K}"
  echo "Max datalake tables: ${SILKMOTH_MAX_DATALAKE_TABLES}"
  echo "Max queries: ${SILKMOTH_MAX_QUERIES}"
  echo "CPU threads: ${CPU_THREADS}"
  echo "Score workers: ${SCORE_WORKERS}"
  echo "Output: ${out_csv}"

  python run_silkmoth_baseline.py \
    --dataset_dir "${dataset_dir}" \
    --query_source "${query_source}" \
    --metric "${SILKMOTH_METRIC}" \
    --relatedness_threshold "${SILKMOTH_THRESHOLD}" \
    --column_name "${SILKMOTH_COLUMN_NAME}" \
    --default_query_column "${SILKMOTH_DEFAULT_QUERY_COLUMN}" \
    --top_k "${SILKMOTH_TOP_K}" \
    --max_datalake_tables "${SILKMOTH_MAX_DATALAKE_TABLES}" \
    --max_queries "${SILKMOTH_MAX_QUERIES}" \
    --score_workers "${SCORE_WORKERS}" \
    --log_level "${SILKMOTH_LOG_LEVEL}" \
    --out_csv "${out_csv}" \
    ${SILKMOTH_EXTRA_ARGS:-}
}

if [[ "${BENCHMARK}" == "all" ]]; then
  for bench in autofj autofj-wdc freyja freyja-wdc wt wt-wdc; do
    run_benchmark "${bench}"
  done
else
  run_benchmark "${BENCHMARK}"
fi
