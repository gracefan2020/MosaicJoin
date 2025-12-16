#!/usr/bin/env python3
"""
Slurm-based Query Processing Script
Submits query processing jobs using Slurm with chunking support.
"""

import os
import subprocess
import math
from pathlib import Path
import pandas as pd

def main():
    # Configuration
    top_k_return = 10
    similarity_threshold = 0.1
    # Value-level entity-link threshold (used only when saving entity links)
    # Keep retrieval permissive (similarity_threshold) but make entity links stricter.
    value_match_threshold = 0.7
    sketch_size = 1024
    queries_per_job = 10
    
    # datalake_dir = "datasets/freyja-semantic-join/datalake"
    # sketches_dir = f"offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/freyja-semantic-join/freyja_query_columns.csv"
    # embeddings_dir = "offline_data/embeddings"
    # output_dir = f"query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"

    datalake_dir = "datasets/autofj_join_benchmark/datalake"
    sketches_dir = f"autofj_offline_data/sketches_k{sketch_size}"
    query_file = "datasets/autofj_join_benchmark/autofj_query_columns.csv"
    embeddings_dir = "autofj_offline_data/embeddings"
    output_dir = f"autofj_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    
    # Validate query file
    if not Path(query_file).exists():
        print(f"❌ Error: Query file not found: {query_file}")
        return 1
    
    # Count queries and calculate jobs
    num_queries = len(pd.read_csv(query_file))
    num_jobs = math.ceil(num_queries / queries_per_job)
    
    print(f"Found {num_queries} queries, creating {num_jobs} jobs")
    
    # Create output directory and script
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    slurm_script = Path(output_dir) / "run_slurm_jobs.sh"
    
    # Generate Slurm batch script
    script_content = f"""#!/bin/bash
#SBATCH --job-name=semantic_query
#SBATCH --output={output_dir}/slurm_%a.out
#SBATCH --error={output_dir}/slurm_%a.err
#SBATCH --array=0-{num_jobs-1}
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

START_INDEX=$((SLURM_ARRAY_TASK_ID * {queries_per_job}))
END_INDEX=$((START_INDEX + {queries_per_job} - 1))
if [ $END_INDEX -ge {num_queries} ]; then
    END_INDEX=$(({num_queries} - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

python run_query_processing.py "{datalake_dir}" "{sketches_dir}" "{query_file}" \\
    --output-dir "{output_dir}/job_$SLURM_ARRAY_TASK_ID" \\
    --top-k-return {top_k_return} \\
    --similarity-threshold {similarity_threshold} \\
    --value-match-threshold {value_match_threshold} \\
    --sketch-size {sketch_size} \\
    --device auto \\
    --embeddings-dir "{embeddings_dir}" \\
    --query-indices ${{QUERY_INDICES}} \\
    --save-entity-links
"""
    
    with open(slurm_script, 'w') as f:
        f.write(script_content)
    os.chmod(slurm_script, 0o755)
    
    print(f"📝 SLURM script created: {slurm_script}")
    print(f"\nTo submit: sbatch {slurm_script}")
    print(f"To check status: squeue -u $USER")
    
    if input("\nSubmit jobs now? (y/n): ").lower() == 'y':
        result = subprocess.run(f"sbatch {slurm_script}", shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Jobs submitted successfully!")
        else:
            print(f"❌ Error: {result.stderr}")
    
    return 0

if __name__ == "__main__":
    exit(main())

