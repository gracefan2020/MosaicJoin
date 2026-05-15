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

SCRIPT_DIR="/scratch/yfw215/MosaicJoin/baselines/topjoin"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/penv"
PYTHON="${ENV_PREFIX}/bin/python"

BENCHMARK="${1:-all}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/baselines/deepjoin/all-mpnet-base-v2}"
TOP_K="${TOPJOIN_TOP_K:-50}"
CANDIDATE_K="${TOPJOIN_CANDIDATE_K:-100}"

module purge
module load anaconda3/2025.06

export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r

echo "=== JOB CHECK ==="
hostname
nvidia-smi

echo "=== ENV CHECK ==="
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "ENV_PREFIX=${ENV_PREFIX}"
echo "PYTHON=${PYTHON}"
which python
"${PYTHON}" -V

cd "${SCRIPT_DIR}"

run_benchmark () {
  local bench="$1"
  local dataset_dir=""
  local query_file=""
  local out_csv=""

  case "${bench}" in
    autofj|autofj-wdc|freyja|freyja-wdc|wt|wt-wdc)
      dataset_dir="${REPO_ROOT}/datasets/${bench}"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/topjoin_ft_${bench}.csv"
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
  echo "Model: ${MODEL_PATH}"
  echo "Output: ${out_csv}"

  "${PYTHON}" infer_full_dataset.py \
    --datalake_dir "${dataset_dir}" \
    --query_file "${query_file}" \
    --out_csv "${out_csv}" \
    --model "${MODEL_PATH}" \
    --benchmark "${bench}" \
    --top_k "${TOP_K}" \
    --candidate_k "${CANDIDATE_K}"
}

if [[ "${BENCHMARK}" == "all" ]]; then
  for bench in autofj autofj-wdc freyja freyja-wdc wt wt-wdc; do
    run_benchmark "${bench}"
  done
else
  run_benchmark "${BENCHMARK}"
fi
