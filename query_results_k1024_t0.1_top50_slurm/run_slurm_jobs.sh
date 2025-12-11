#!/bin/bash
#SBATCH --job-name=semantic_query
#SBATCH --output=query_results_k1024_t0.1_top50_slurm/slurm_%a.out
#SBATCH --error=/dev/null
#SBATCH --array=0-4
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Calculate query range for this job
START_INDEX=$((SLURM_ARRAY_TASK_ID * 10))
END_INDEX=$((START_INDEX + 10 - 1))

# Adjust END_INDEX if it exceeds the number of queries
if [ $END_INDEX -ge 50 ]; then
    END_INDEX=$((50 - 1))
fi

echo "Job $SLURM_ARRAY_TASK_ID: Processing queries $START_INDEX to $END_INDEX"
echo "Total queries in job: $((END_INDEX - START_INDEX + 1))"

# Build query indices list
QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

# Run query processing with specific query indices
python run_query_processing.py "datasets/freyja-semantic-join/datalake" "offline_data/sketches_k1024" "datasets/freyja-semantic-join/freyja_query_columns.csv" \
    --output-dir "query_results_k1024_t0.1_top50_slurm/job_$SLURM_ARRAY_TASK_ID" \
    --top-k-return 50 \
    --similarity-threshold 0.1 \
    --sketch-size 1024 \
    --device "auto" \
    --embeddings-dir "offline_data/embeddings" \
    --query-indices ${QUERY_INDICES}

echo "Job $SLURM_ARRAY_TASK_ID completed successfully"
