#!/bin/bash
#SBATCH --job-name=semantic_query
#SBATCH --output=wt-experiments/wt_query_results_k1024_t0.1_top50_slurm_no_column_names/slurm_%a.out
#SBATCH --error=wt-experiments/wt_query_results_k1024_t0.1_top50_slurm_no_column_names/slurm_%a.err
#SBATCH --array=0-11
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

START_INDEX=$((SLURM_ARRAY_TASK_ID * 10))
END_INDEX=$((START_INDEX + 10 - 1))
if [ $END_INDEX -ge 119 ]; then
    END_INDEX=$((119 - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

python run_query_processing.py "datasets/wt/datalake_no_column_names" "wt-experiments/wt_offline_data_no_column_names/sketches_k1024" "datasets/wt/wt_query_columns_no_column_names.csv" \
    --output-dir "wt-experiments/wt_query_results_k1024_t0.1_top50_slurm_no_column_names/job_$SLURM_ARRAY_TASK_ID" \
    --top-k-return 50 \
    --similarity-threshold 0.1 \
    --similarity-method "greedy_match" \
    --sketch-size 1024 \
    --device auto \
    --embeddings-dir "wt-experiments/wt_offline_data_no_column_names/embeddings" \
    --query-indices ${QUERY_INDICES}
