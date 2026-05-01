#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --account=torch_pr_66_tandon_advanced
#SBATCH --comment="preemption=yes;requeue=true"

set -eo pipefail

SCRIPT_DIR="/scratch/yfw215/MosaicJoin/baselines/koios"
ENV_PREFIX="${KOIOS_ENV_PREFIX:-${SCRIPT_DIR}/penv}"
PYTHON="${ENV_PREFIX}/bin/python"

BENCHMARK="${1:-all}"
TOP_K="${KOIOS_TOP_K:-50}"
ALPHA="${KOIOS_ALPHA:-0.7}"
MAX_VALUES_PER_SET="${KOIOS_MAX_VALUES_PER_SET:-128}"
CANDIDATE_MODE="${KOIOS_CANDIDATE_MODE:-faiss}"
FAISS_DEVICE="${KOIOS_FAISS_DEVICE:-gpu}"
FAISS_GPU_ID="${KOIOS_FAISS_GPU_ID:-0}"
FAISS_K="${KOIOS_FAISS_K:-2048}"
CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"
SCORE_WORKERS="${KOIOS_SCORE_WORKERS:-${CPU_THREADS}}"

module purge
module load anaconda3/2025.06

export PATH="${ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
hash -r

echo "=== JOB CHECK ==="
hostname
nvidia-smi

echo "=== ENV CHECK ==="
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "ENV_PREFIX=${ENV_PREFIX}"
echo "PYTHON=${PYTHON}"
echo "CPU_THREADS=${CPU_THREADS}"
echo "SCORE_WORKERS=${SCORE_WORKERS}"
which python
"${PYTHON}" -V
"${PYTHON}" -c "import sys; print(sys.executable); import faiss; print(faiss.__version__); print('gpus:', faiss.get_num_gpus())"

cd "${SCRIPT_DIR}"

run_benchmark () {
  local bench="$1"
  local work_dir="${SCRIPT_DIR}/work/${bench}"
  local sets_dir="${work_dir}/sets"
  local db_path="${work_dir}/ft.sqlite3"
  local manifest_json="${work_dir}/sets_manifest.json"
  local out_csv="${SCRIPT_DIR}/koios_ft_${bench}.csv"

  case "${bench}" in
    autofj|autofj-wdc|freyja|freyja-wdc|wt|wt-wdc)
      ;;
    *)
      echo "Unsupported benchmark: ${bench}"
      echo "Supported: all, autofj, autofj-wdc, freyja, freyja-wdc, wt, wt-wdc"
      return 1
      ;;
  esac

  echo "=============================================="
  echo "Benchmark: ${bench}"
  echo "Sets: ${sets_dir}"
  echo "DB: ${db_path}"
  echo "Output: ${out_csv}"
  echo "Top-k: ${TOP_K}"
  echo "Alpha: ${ALPHA}"
  echo "Max values per set at query time: ${MAX_VALUES_PER_SET}"

  if [[ ! -f "${db_path}" ]]; then
    echo "Missing ${db_path}. Build it first with build_koios_embedding_db.py or build_embeddings.SBATCH."
    return 1
  fi

  "${PYTHON}" run_koios_baseline.py \
    --dataset_name "${bench}" \
    --sets_dir "${sets_dir}" \
    --db "${db_path}" \
    --manifest_json "${manifest_json}" \
    --top_k "${TOP_K}" \
    --alpha "${ALPHA}" \
    --max_values_per_set "${MAX_VALUES_PER_SET}" \
    --candidate_mode "${CANDIDATE_MODE}" \
    --faiss_device "${FAISS_DEVICE}" \
    --faiss_gpu_id "${FAISS_GPU_ID}" \
    --faiss_k "${FAISS_K}" \
    --score_workers "${SCORE_WORKERS}" \
    --out_csv "${out_csv}" \
    ${KOIOS_EXTRA_ARGS:-}
}

if [[ "${BENCHMARK}" == "all" ]]; then
  for bench in autofj autofj-wdc freyja freyja-wdc wt wt-wdc; do
    run_benchmark "${bench}"
  done
else
  run_benchmark "${BENCHMARK}"
fi
