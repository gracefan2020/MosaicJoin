#!/bin/bash
#SBATCH --job-name=llm_entity_linking
#SBATCH --output=entity-linking-experiments/autofj_query_results_k1024_t0.1_top10_slurm/llm_entity_linking_%a.out
#SBATCH --error=entity-linking-experiments/autofj_query_results_k1024_t0.1_top10_slurm/llm_entity_linking_%a.err
#SBATCH --array=0-4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

START_INDEX=$((SLURM_ARRAY_TASK_ID * 100))
END_INDEX=$((START_INDEX + 100 - 1))
if [ $END_INDEX -ge 500 ]; then
    END_INDEX=$((500 - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

cd /scratch/gf2467/SemSketch
python entity-linking-experiments/llm_entity_linking.py "entity-linking-experiments/autofj_query_results_k1024_t0.1_top10_slurm" --query-indices ${QUERY_INDICES}
