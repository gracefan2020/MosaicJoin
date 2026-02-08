#!/bin/bash
#SBATCH --job-name=semantic_query
#SBATCH --output=wt-experiments/wt_query_results_embeddinggemma_D128_Q64_chamfer_top50/slurm_%a.out
#SBATCH --error=wt-experiments/wt_query_results_embeddinggemma_D128_Q64_chamfer_top50/slurm_%a.err
#SBATCH --array=0-11
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_tandon

START_INDEX=$((SLURM_ARRAY_TASK_ID * 10))
END_INDEX=$((START_INDEX + 10 - 1))
if [ $END_INDEX -ge 119 ]; then
    END_INDEX=$((119 - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

python run_query_processing.py "datasets/wt/datalake_no_column_names" "wt-experiments/wt_offline_data_embeddinggemma_no_column_names/sketches_k128" "datasets/wt/wt_query_columns_no_column_names.csv" \
    --output-dir "wt-experiments/wt_query_results_embeddinggemma_D128_Q64_chamfer_top50/job_$SLURM_ARRAY_TASK_ID" \
    --top-k-return 50 \
    --similarity-threshold 0.1 \
    --similarity-method "chamfer" \
    --sketch-size 64 \
    --device cuda \
    --embeddings-dir "wt-experiments/wt_offline_data_{embedding_model}_no_column_names/embeddings" \
    --embedding-model "embeddinggemma" \
    --embedding-dim 128 \
    --query-indices ${QUERY_INDICES}
