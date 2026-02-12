#!/bin/bash
#SBATCH --job-name=semantic_query
#SBATCH --output=entity-linking-experiments/autofj_query_results_k1024_t0.1_top10_slurm/slurm_%a.out
#SBATCH --error=entity-linking-experiments/autofj_query_results_k1024_t0.1_top10_slurm/slurm_%a.err
#SBATCH --array=0-4
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

START_INDEX=$((SLURM_ARRAY_TASK_ID * 10))
END_INDEX=$((START_INDEX + 10 - 1))
if [ $END_INDEX -ge 50 ]; then
    END_INDEX=$((50 - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

python run_query_processing.py "datasets/autofj_join_benchmark/datalake" "entity-linking-experiments/autofj_offline_data/sketches_k1024" "datasets/autofj_join_benchmark/autofj_query_columns.csv" \
    --output-dir "entity-linking-experiments/autofj_query_results_k1024_t0.1_top10_slurm/job_$SLURM_ARRAY_TASK_ID" \
    --top-k-return 10 \
    --similarity-threshold 0.1 \
    --similarity-method "greedy_match" \
    --sketch-size 1024 \
    --device auto \
    --embeddings-dir "entity-linking-experiments/autofj_offline_data/embeddings" \
    --query-indices ${QUERY_INDICES}
