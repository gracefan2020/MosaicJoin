#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem 128GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --account=torch_pr_66_tandon_advanced
#SBATCH --comment="preemption=yes;requeue=true"

BENCHMARK="${1:-wt-wdc}"
BATCH_SIZE="${KOIOS_BUILD_BATCH_SIZE:-1024}"
DEVICE="${KOIOS_BUILD_DEVICE:-cuda}"

cd /scratch/yfw215/MosaicJoin/baselines/koios

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate ./penv

nvidia-smi

python build_koios_embedding_db.py \
  --dataset_name "${BENCHMARK}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --overwrite
