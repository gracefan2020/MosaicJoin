#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem 128GB
#SBATCH --time 12:00:00
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
MODEL_PATH="${SCRIPT_DIR}/all-mpnet-base-v2"

module purge
module load anaconda3/2025.06

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

echo "=== GPU CHECK ==="
nvidia-smi || true

cd "${SCRIPT_DIR}"

run_benchmark () {
  local bench="$1"
  local dataset_dir=""
  local query_file=""
  local out_csv=""

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
      echo "Supported: all, autofj, autofj-wdc, freyja, freyja-wdc, wt, wt-wdc"
      return 1
      ;;
  esac

  echo "=============================================="
  echo "Benchmark: ${bench}"
  echo "Dataset: ${dataset_dir}"
  echo "Query file: ${query_file}"
  echo "Output: ${out_csv}"

  python infer_full_dataset.py \
    --datalake_dir "${dataset_dir}" \
    --model_name "${MODEL_PATH}" \
    --out_csv "${out_csv}" \
    --with_header \
    --query_dir "${query_file}" \
    --device auto
}

if [[ "${BENCHMARK}" == "all" ]]; then
  for bench in autofj autofj-wdc freyja freyja-wdc wt wt-wdc; do
    run_benchmark "${bench}"
  done
else
  run_benchmark "${BENCHMARK}"
fi
