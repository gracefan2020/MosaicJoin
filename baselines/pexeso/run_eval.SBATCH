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
SMALL_THRESHOLD="0.01"

module purge
module load anaconda3/2025.06

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

echo "=== GPU CHECK ==="
nvidia-smi || true

cd "${SCRIPT_DIR}"

run_benchmark () {
  local bench="$1"
  local embedding_mode="$2"
  local dataset_dir=""
  local query_source=""
  local out_csv=""
  local index_cache=""

  case "${bench}" in
    autofj)
      dataset_dir="${REPO_ROOT}/datasets/autofj"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/deepjoin_ft_autofj.csv"
      ;;
    autofj-wdc)
      dataset_dir="${REPO_ROOT}/datasets/autofj-wdc"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/deepjoin_ft_autofj-wdc.csv"
      ;;
    freyja)
      dataset_dir="${REPO_ROOT}/datasets/freyja"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/deepjoin_ft_freyja.csv"
      ;;
    freyja-wdc)
      dataset_dir="${REPO_ROOT}/datasets/freyja-wdc"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/deepjoin_ft_freyja-wdc.csv"
      ;;
    wt)
      dataset_dir="${REPO_ROOT}/datasets/wt"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/deepjoin_ft_wt.csv"
      ;;
    wt-wdc)
      dataset_dir="${REPO_ROOT}/datasets/wt-wdc"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/deepjoin_ft_wt-wdc.csv"
      ;;
    *)
      echo "Unsupported benchmark: ${bench}"
      echo "Supported: all, autofj-wdc, autofj_join_benchmark, freyja-semantic-join, gdc-breakdown, wt"
      return 1
      ;;
  esac

  echo "=============================================="
  echo "Benchmark: ${bench}"
  echo "Dataset: ${dataset_dir}"
  echo "Query source: ${query_source}"
  echo "Output: ${out_csv}"
  echo "Containment threshold: ${SMALL_THRESHOLD}"

  mkdir -p "${SCRIPT_DIR}/cache"
  index_cache="${SCRIPT_DIR}/cache/pexeso-${bench}-${embedding_mode}.pkl"
  echo "Index cache: ${index_cache}"
  rm -f "${index_cache}"
  echo "Removed old cache for this task (if any)"

  python run_pexeso_baseline.py \
    --dataset_dir "${dataset_dir}" \
    --query_source "${query_source}" \
    --embedding_mode "${embedding_mode}" \
    --device auto \
    --tau 0.6 \
    --containment_threshold "${SMALL_THRESHOLD}" \
    --time_threshold 600 \
    --log_level INFO \
    --index_cache "${index_cache}" \
    --out_csv "${out_csv}"
}

if [[ "${BENCHMARK}" == "all" ]]; then
  for embedding_mode in "fasttext" "mpnet"; do
    echo "Running benchmarks with embedding mode: ${embedding_mode}"
    for bench in autofj autofj-wdc freyja freyja-wdc wt wt-wdc; do
      run_benchmark "${bench}" "${embedding_mode}"
    done
  done
else
  run_benchmark "${BENCHMARK}" "fasttext"
fi
