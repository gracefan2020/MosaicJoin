#!/bin/bash
#SBATCH --job-name=llm_postprocessing
#SBATCH --output=llm_postprocessing_%j.out
#SBATCH --error=llm_postprocessing_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Default query results directory
QUERY_RESULTS_DIR="${1:-query_results_k1024_t0.7_top50_deepjoin_N100_K500_T0.6}"
# QUERY_RESULTS_DIR="${1:-query_results_k1024_t0.7_top50}"


echo "Starting LLM postprocessing for: $QUERY_RESULTS_DIR"
echo "Start time: $(date)"


# Run the script
python llm_postprocessing.py "$QUERY_RESULTS_DIR"

echo "End time: $(date)"
echo "Job completed"

