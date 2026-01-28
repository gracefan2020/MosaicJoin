#!/bin/bash
#SBATCH --job-name=autofj_baseline
#SBATCH --array=0-9
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

START_INDEX=$((SLURM_ARRAY_TASK_ID * 50))
END_INDEX=$((START_INDEX + 50 - 1))
if [ $END_INDEX -ge 500 ]; then
    END_INDEX=$((500 - 1))
fi

cd "/scratch/gf2467/SemSketch/entity-linking-experiments" || exit 1

python "/scratch/gf2467/SemSketch/entity-linking-experiments/autofj_deepjoin_baseline.py" \
    "/scratch/gf2467/SemSketch/Deepjoin/output-autofj/deepjoin_results_K10_N10_T0.1.csv" \
    "/scratch/gf2467/SemSketch/datasets/autofj_join_benchmark/datalake" \
    "/scratch/gf2467/SemSketch/entity-linking-experiments/autofj_deepjoin_baseline_k10_n10_t0.1/job_$SLURM_ARRAY_TASK_ID" \
    "0.9" \
    --start-index $START_INDEX \
    --end-index $END_INDEX \
    --pairs-file "/scratch/gf2467/SemSketch/entity-linking-experiments/autofj_deepjoin_baseline_k10_n10_t0.1/unique_pairs.csv" \
    --query-columns "/scratch/gf2467/SemSketch/datasets/autofj_join_benchmark/autofj_query_columns.csv"
