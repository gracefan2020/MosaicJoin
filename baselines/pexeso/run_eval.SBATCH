#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
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
SMALL_THRESHOLD="${SMALL_THRESHOLD:-0}"
NUM_PIVOTS="${NUM_PIVOTS:-3}"
NUM_LAYERS="${NUM_LAYERS:-4}"
TIME_THRESHOLD="${TIME_THRESHOLD:-60}"
MAX_UNIQUE_VALUES="${MAX_UNIQUE_VALUES:-0}"
MAX_INDEX_UNIQUE_VALUES="${MAX_INDEX_UNIQUE_VALUES:-256}"
MAX_QUERY_UNIQUE_VALUES="${MAX_QUERY_UNIQUE_VALUES:-128}"

module purge
module load anaconda3/2025.06

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

echo "=== GPU CHECK ==="
nvidia-smi || true

# Optional keep-alive for clusters that terminate low-GPU-util jobs.
# Toggle with: export GPU_KEEPALIVE_ENABLE=1
GPU_KEEPALIVE_ENABLE="${GPU_KEEPALIVE_ENABLE:-0}"
GPU_KEEPALIVE_INTERVAL_SEC="${GPU_KEEPALIVE_INTERVAL_SEC:-5}"
GPU_KEEPALIVE_WORK_MS="${GPU_KEEPALIVE_WORK_MS:-4500}"
GPU_KEEPALIVE_MATRIX_SIZE="${GPU_KEEPALIVE_MATRIX_SIZE:-4096}"
GPU_KEEPALIVE_TARGET_MEM_FRAC="${GPU_KEEPALIVE_TARGET_MEM_FRAC:-0.55}"
GPU_KEEPALIVE_RESERVE_MEM_GB="${GPU_KEEPALIVE_RESERVE_MEM_GB:-8}"
GPU_KEEPALIVE_LOG="${SCRIPT_DIR}/gpu_keepalive-${SLURM_JOB_ID:-manual}.log"
KEEPALIVE_PID=""

start_gpu_keepalive () {
  if [[ "${GPU_KEEPALIVE_ENABLE}" != "1" ]]; then
    echo "GPU keep-alive disabled (GPU_KEEPALIVE_ENABLE=${GPU_KEEPALIVE_ENABLE})"
    return 0
  fi
  if ! python -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null | grep -q "^1$"; then
    echo "GPU keep-alive not started (CUDA unavailable)"
    return 0
  fi

  echo "Starting GPU keep-alive: interval=${GPU_KEEPALIVE_INTERVAL_SEC}s work=${GPU_KEEPALIVE_WORK_MS}ms matrix=${GPU_KEEPALIVE_MATRIX_SIZE} mem_frac=${GPU_KEEPALIVE_TARGET_MEM_FRAC}"
  python "${SCRIPT_DIR}/gpu_keepalive.py" \
    --device cuda \
    --interval_sec "${GPU_KEEPALIVE_INTERVAL_SEC}" \
    --work_ms "${GPU_KEEPALIVE_WORK_MS}" \
    --matrix_size "${GPU_KEEPALIVE_MATRIX_SIZE}" \
    --target_mem_frac "${GPU_KEEPALIVE_TARGET_MEM_FRAC}" \
    --reserve_mem_gb "${GPU_KEEPALIVE_RESERVE_MEM_GB}" \
    > "${GPU_KEEPALIVE_LOG}" 2>&1 &
  KEEPALIVE_PID=$!
  echo "GPU keep-alive PID=${KEEPALIVE_PID}, log=${GPU_KEEPALIVE_LOG}"
}

stop_gpu_keepalive () {
  if [[ -n "${KEEPALIVE_PID}" ]]; then
    if kill -0 "${KEEPALIVE_PID}" 2>/dev/null; then
      echo "Stopping GPU keep-alive PID=${KEEPALIVE_PID}"
      kill "${KEEPALIVE_PID}" 2>/dev/null || true
      wait "${KEEPALIVE_PID}" 2>/dev/null || true
    fi
  fi
}

trap stop_gpu_keepalive EXIT INT TERM

cd "${SCRIPT_DIR}"
start_gpu_keepalive

run_benchmark () {
  local bench="$1"
  local embedding_mode="$2"
  local dataset_dir=""
  local query_source=""
  local query_file=""
  local out_csv=""
  local index_cache=""

  case "${bench}" in
    autofj)
      dataset_dir="${REPO_ROOT}/datasets/autofj"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/pexeso_${embedding_mode}_autofj.csv"
      ;;
    autofj-wdc)
      dataset_dir="${REPO_ROOT}/datasets/autofj-wdc"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/pexeso_${embedding_mode}_autofj-wdc.csv"
      ;;
    freyja)
      dataset_dir="${REPO_ROOT}/datasets/freyja"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/pexeso_${embedding_mode}_freyja.csv"
      ;;
    freyja-wdc)
      dataset_dir="${REPO_ROOT}/datasets/freyja-wdc"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/pexeso_${embedding_mode}_freyja-wdc.csv"
      ;;
    wt)
      dataset_dir="${REPO_ROOT}/datasets/wt"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/pexeso_${embedding_mode}_wt.csv"
      ;;
    wt-wdc)
      dataset_dir="${REPO_ROOT}/datasets/wt-wdc"
      query_file="${dataset_dir}/query_columns.csv"
      out_csv="${SCRIPT_DIR}/pexeso_${embedding_mode}_wt-wdc.csv"
      ;;
    *)
      echo "Unsupported benchmark: ${bench}"
      echo "Supported: all, autofj-wdc, autofj_join_benchmark, freyja-semantic-join, gdc-breakdown, wt"
      return 1
      ;;
  esac

  if [[ -n "${query_file}" ]]; then
    query_source="${query_file}"
  fi

  echo "=============================================="
  echo "Benchmark: ${bench}"
  echo "Dataset: ${dataset_dir}"
  echo "Query source: ${query_source}"
  echo "Output: ${out_csv}"
  echo "Containment threshold: ${SMALL_THRESHOLD}"
  echo "Num pivots/layers: ${NUM_PIVOTS}/${NUM_LAYERS}"
  echo "Per-query verify timeout: ${TIME_THRESHOLD}s"
  echo "Sampling caps: max_unique_values=${MAX_UNIQUE_VALUES}, max_index_unique_values=${MAX_INDEX_UNIQUE_VALUES}, max_query_unique_values=${MAX_QUERY_UNIQUE_VALUES}"

  mkdir -p "${SCRIPT_DIR}/cache"
  index_cache="${SCRIPT_DIR}/cache/pexeso-${bench}-${embedding_mode}.pkl"
  echo "Index cache: ${index_cache}"
  if [[ "${REBUILD_CACHE:-0}" == "1" ]]; then
    rm -f "${index_cache}"
    echo "Removed old cache for this task (REBUILD_CACHE=1)"
  fi

  python run_pexeso_baseline.py \
    --dataset_dir "${dataset_dir}" \
    --query_source "${query_source}" \
    --embedding_mode "${embedding_mode}" \
    --device auto \
    --num_pivots "${NUM_PIVOTS}" \
    --num_layers "${NUM_LAYERS}" \
    --pivot_method lof \
    --tau 0.6 \
    --containment_threshold "${SMALL_THRESHOLD}" \
    --time_threshold "${TIME_THRESHOLD}" \
    --score_mode containment \
    --prune_multiplier 2.0 \
    --max_unique_values "${MAX_UNIQUE_VALUES}" \
    --max_index_unique_values "${MAX_INDEX_UNIQUE_VALUES}" \
    --max_query_unique_values "${MAX_QUERY_UNIQUE_VALUES}" \
    --log_level INFO \
    --index_cache "${index_cache}" \
    --out_csv "${out_csv}"
}

if [[ "${BENCHMARK}" == "all" ]]; then
  for embedding_mode in "fasttext"; do
    echo "Running benchmarks with embedding mode: ${embedding_mode}"
    for bench in freyja wt autofj-wdc freyja-wdc wt-wdc; do
      run_benchmark "${bench}" "${embedding_mode}"
    done
  done
else
  run_benchmark "${BENCHMARK}" "fasttext"
fi
